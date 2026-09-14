import pycuda.driver as cuda
import pycuda.autoinit
from cuda.bindings import nvrtc
import numpy as np
import time

FNV_PRIME = 0x01000193
FNV_OFFSET_BASIS = 0x811c9dc5

def fnv1a(u, v):
    return ((u ^ v) * FNV_PRIME) & 0xFFFFFFFF

class KISS99:
    def __init__(self, z, w, jsr, jcong):
        self.z = z & 0xFFFFFFFF
        self.w = w & 0xFFFFFFFF
        self.jsr = jsr & 0xFFFFFFFF
        self.jcong = jcong & 0xFFFFFFFF

    def __call__(self):
        self.z = (36969 * (self.z & 65535) + (self.z >> 16)) & 0xFFFFFFFF
        self.w = (18000 * (self.w & 65535) + (self.w >> 16)) & 0xFFFFFFFF
        mwc = ((self.z << 16) + self.w) & 0xFFFFFFFF
        self.jsr ^= (self.jsr << 17) & 0xFFFFFFFF
        self.jsr ^= (self.jsr >> 13) & 0xFFFFFFFF
        self.jsr ^= (self.jsr << 5) & 0xFFFFFFFF
        self.jcong = (69069 * self.jcong + 1234567) & 0xFFFFFFFF
        return (((mwc ^ self.jcong) + self.jsr)) & 0xFFFFFFFF

def gen_merge_expr(target_expr, val_expr, selector):
    s = selector % 4
    x = ((selector >> 16) % 31) + 1
    if s == 0:
        return f"{target_expr} = ({target_expr} * 33) + ({val_expr});"
    elif s == 1:
        return f"{target_expr} = ({target_expr} ^ ({val_expr})) * 33;"
    elif s == 2:
        return f"{target_expr} = ROTL32({target_expr}, {x}) ^ ({val_expr});"
    elif s == 3:
        return f"{target_expr} = ROTR32({target_expr}, {x}) ^ ({val_expr});"

def gen_math_expr(a_expr, b_expr, selector):
    s = selector % 11
    if s == 0: return f"({a_expr} + {b_expr})"
    if s == 1: return f"({a_expr} * {b_expr})"
    if s == 2: return f"(uint32_t)(((uint64_t)({a_expr}) * (uint64_t)({b_expr})) >> 32)"
    if s == 3: return f"(({a_expr} < {b_expr}) ? {a_expr} : {b_expr})"
    if s == 4: return f"ROTL32({a_expr}, ({b_expr}) & 31)"
    if s == 5: return f"ROTR32({a_expr}, ({b_expr}) & 31)"
    if s == 6: return f"({a_expr} & {b_expr})"
    if s == 7: return f"({a_expr} | {b_expr})"
    if s == 8: return f"({a_expr} ^ {b_expr})"
    if s == 9: return f"(__clz({a_expr}) + __clz({b_expr}))"
    if s == 10: return f"(__popc({a_expr}) + __popc({b_expr}))"

def generate_round_code(period):
    p_lo = period & 0xFFFFFFFF
    p_hi = period >> 32
    r_z = fnv1a(FNV_OFFSET_BASIS, p_lo)
    r_w = fnv1a(r_z, p_hi)
    r_jsr = fnv1a(r_w, p_lo)
    r_jcong = fnv1a(r_jsr, p_hi)
    rng = KISS99(r_z, r_w, r_jsr, r_jcong)

    dst_seq = list(range(32))
    src_seq = list(range(32))
    for i in range(32, 1, -1):
        r1 = rng() % i
        dst_seq[i - 1], dst_seq[r1] = dst_seq[r1], dst_seq[i - 1]
        r2 = rng() % i
        src_seq[i - 1], src_seq[r2] = src_seq[r2], src_seq[i - 1]

    dst_counter = 0
    src_counter = 0

    lines = []
    for i in range(18):
        if i < 11:
            src = src_seq[src_counter % 32]; src_counter += 1
            dst = dst_seq[dst_counter % 32]; dst_counter += 1
            sel = rng()
            val = f"shared_l1[mix[{src}] % 4096]"
            lines.append(gen_merge_expr(f"mix[{dst}]", val, sel))
        if i < 18:
            src_rnd = rng() % (32 * 31)
            src1 = src_rnd % 32
            src2 = src_rnd // 32
            if src2 >= src1: src2 += 1
            sel1 = rng()
            dst = dst_seq[dst_counter % 32]; dst_counter += 1
            sel2 = rng()
            math_expr = gen_math_expr(f"mix[{src1}]", f"mix[{src2}]", sel1)
            lines.append(gen_merge_expr(f"mix[{dst}]", math_expr, sel2))

    dag_dsts = []
    dag_sels = []
    for i in range(4):
        d = 0 if i == 0 else dst_seq[dst_counter % 32]
        if i != 0: dst_counter += 1
        s = rng()
        dag_dsts.append(d)
        dag_sels.append(s)

    dag_code = []
    for i in range(4):
        val = f"__ldg(&dag[dag_word_offset + {i}])"
        dag_code.append(gen_merge_expr(f"mix[{dag_dsts[i]}]", val, dag_sels[i]))

    return "\n        ".join(lines), "\n        ".join(dag_code)

def build_jit_kernel_source(period):
    round_math, round_dag = generate_round_code(period)
    return r"""
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
#define ROTR32(x, r) (((x) >> (r)) | ((x) << (32 - (r))))
#define FNV_PRIME 0x01000193u
#define FNV_OFFSET_BASIS 0x811c9dc5u

__device__ __forceinline__ uint32_t fnv1a(uint32_t u, uint32_t v) {
    return (u ^ v) * FNV_PRIME;
}

__constant__ uint32_t RC800[22] = {
    0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001,
    0x80008081, 0x00008009, 0x0000008A, 0x00000088, 0x80008009, 0x8000000A,
    0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800A, 0x8000000A, 0x80008081, 0x00008080
};

__constant__ uint8_t ROT800[5][5] = {
    {0, 4, 3, 9, 18},
    {1, 12, 10, 13, 2},
    {30, 6, 11, 15, 29},
    {28, 23, 25, 21, 24},
    {27, 20, 7, 8, 14}
};

__constant__ uint32_t RAVENCOIN_KAWPOW[15] = {
    0x72, 0x41, 0x56, 0x45, 0x4E, 0x43, 0x4F, 0x49, 0x4E, 0x4B, 0x41, 0x57, 0x50, 0x4F, 0x57
};

__device__ void keccak_f800(uint32_t state[25]) {
    uint32_t A[5][5];
    #pragma unroll
    for (int x = 0; x < 5; x++) {
        #pragma unroll
        for (int y = 0; y < 5; y++) {
            A[x][y] = state[x + 5 * y];
        }
    }
    #pragma unroll
    for (int round_idx = 0; round_idx < 22; round_idx++) {
        uint32_t C[5];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            C[x] = A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4];
        }
        uint32_t D[5];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ ROTL32(C[(x + 1) % 5], 1);
        }
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                A[x][y] ^= D[x];
            }
        }
        uint32_t B[5][5];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                B[y][(2 * x + 3 * y) % 5] = ROTL32(A[x][y], ROT800[x][y]);
            }
        }
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]);
            }
        }
        A[0][0] ^= RC800[round_idx];
    }
    #pragma unroll
    for (int x = 0; x < 5; x++) {
        #pragma unroll
        for (int y = 0; y < 5; y++) {
            state[x + 5 * y] = A[x][y];
        }
    }
}

typedef struct {
    uint32_t z, w, jsr, jcong;
} kiss99_t;

__device__ __forceinline__ uint32_t kiss99(kiss99_t *st) {
    st->z = 36969 * (st->z & 65535) + (st->z >> 16);
    st->w = 18000 * (st->w & 65535) + (st->w >> 16);
    uint32_t mwc = ((st->z << 16) + st->w);
    st->jsr ^= (st->jsr << 17);
    st->jsr ^= (st->jsr >> 13);
    st->jsr ^= (st->jsr << 5);
    st->jcong = 69069 * st->jcong + 1234567;
    return ((mwc ^ st->jcong) + st->jsr);
}

extern "C" __global__ void kawpow_search_jit(const uint32_t *header_words, uint64_t start_nonce, uint64_t target_high64,
                                             const uint32_t *dag, uint32_t num_dag_items_2048,
                                             uint32_t batch_size, uint64_t *found_nonces, uint8_t *found_mixes,
                                             uint32_t *found_count) {
    __shared__ uint32_t shared_l1[4096];
    for (int idx = threadIdx.x; idx < 4096; idx += blockDim.x) {
        shared_l1[idx] = dag[idx];
    }
    __syncthreads();

    uint32_t lane_id = threadIdx.x % 16;
    uint32_t group_id = ((blockIdx.x * blockDim.x) + threadIdx.x) / 16;
    if (group_id >= batch_size) return;
    uint64_t nonce = start_nonce + group_id;

    // 1. Initial Keccak
    uint32_t state[25];
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) state[i] = header_words[i];
        state[8] = (uint32_t)(nonce & 0xFFFFFFFFULL);
        state[9] = (uint32_t)(nonce >> 32);
        #pragma unroll
        for (int i = 10; i < 25; i++) state[i] = RAVENCOIN_KAWPOW[i - 10];
        keccak_f800(state);
    }

    uint32_t state2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state2[i] = __shfl_sync(0xffff, state[i], 0);
    }

    // 2. Mix init
    uint32_t z = fnv1a(FNV_OFFSET_BASIS, state2[0]);
    uint32_t w = fnv1a(z, state2[1]);
    uint32_t jsr = fnv1a(w, lane_id);
    uint32_t jcong = fnv1a(jsr, lane_id);
    kiss99_t my_rng = {z, w, jsr, jcong};

    uint32_t mix[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        mix[i] = kiss99(&my_rng);
    }

    // 3. 64-round unrolled ProgPOW loop
    for (int r = 0; r < 64; r++) {
        uint32_t mix_r0 = __shfl_sync(0xffff, mix[0], r % 16);
        uint32_t item_index = mix_r0 % num_dag_items_2048;

        // JIT generated straight-line math & cache ops
        """ + round_math + r"""

        // JIT generated DAG loads
        uint32_t dag_word_offset = (item_index * 64) + (((lane_id ^ r) % 16) * 4);
        """ + round_dag + r"""
    }

    // 4. Reduction
    uint32_t lane_hash = FNV_OFFSET_BASIS;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        lane_hash = fnv1a(lane_hash, mix[i]);
    }

    uint32_t mix_hash[8];
    #pragma unroll
    for (int w_idx = 0; w_idx < 8; w_idx++) {
        uint32_t h0 = __shfl_sync(0xffff, lane_hash, w_idx);
        uint32_t h1 = __shfl_sync(0xffff, lane_hash, w_idx + 8);
        mix_hash[w_idx] = fnv1a(fnv1a(FNV_OFFSET_BASIS, h0), h1);
    }

    // 5. Final Keccak & target check
    if (lane_id == 0) {
        uint32_t st_final[25];
        #pragma unroll
        for (int i = 0; i < 8; i++) st_final[i] = state2[i];
        #pragma unroll
        for (int i = 8; i < 16; i++) st_final[i] = mix_hash[i - 8];
        #pragma unroll
        for (int i = 16; i < 25; i++) st_final[i] = RAVENCOIN_KAWPOW[i - 16];
        keccak_f800(st_final);

        uint64_t w0 = ((uint64_t)st_final[1] << 32) | (uint64_t)st_final[0];
        uint64_t hash_val_be = (((w0 & 0x00000000000000ffULL) << 56) |
                                ((w0 & 0x000000000000ff00ULL) << 40) |
                                ((w0 & 0x0000000000ff0000ULL) << 24) |
                                ((w0 & 0x00000000ff000000ULL) <<  8) |
                                ((w0 & 0x000000ff00000000ULL) >>  8) |
                                ((w0 & 0x0000ff0000000000ULL) >> 24) |
                                ((w0 & 0x00ff000000000000ULL) >> 40) |
                                ((w0 & 0xff00000000000000ULL) >> 56));

        if (hash_val_be <= target_high64) {
            uint32_t idx = atomicAdd(found_count, 1);
            if (idx < 1024) {
                found_nonces[idx] = nonce;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    ((uint32_t*)found_mixes)[idx * 8 + b] = mix_hash[b];
                }
            }
        }
    }
}
"""

def compile_jit(period):
    src = build_jit_kernel_source(period)
    dev = cuda.Device(0)
    cc = dev.compute_capability()
    arch = f"sm_{cc[0]}{cc[1]}"

    t0 = time.time()
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"jit.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
    err, = nvrtc.nvrtcCompileProgram(prog, 1, [f"--gpu-architecture={arch}".encode()])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, sz = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * sz
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC JIT compilation failed:\n{log.decode()}")

    _, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * cubin_size
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    mod = cuda.module_from_buffer(cubin)
    print(f"JIT compiled period {period} in {time.time() - t0:.3f}s")
    return mod.get_function("kawpow_search_jit")

if __name__ == '__main__':
    print("Testing JIT Compilation...")
    fn = compile_jit(0)
    print(f"Kernel registers: {fn.num_regs}, max threads/block: {fn.max_threads_per_block}")

    # Allocate dummy 1GB buffer on GPU for epoch 0 DAG
    dag_words = 256 * 1024 * 1024
    dag_gpu = cuda.mem_alloc(dag_words * 4)
    cuda.memset_d32(dag_gpu, 0x12345678, dag_words)

    header_gpu = cuda.mem_alloc(32)
    cuda.memset_d32(header_gpu, 0, 8)

    max_found = 1024
    found_nonces_gpu = cuda.mem_alloc(max_found * 8)
    found_mixes_gpu = cuda.mem_alloc(max_found * 32)
    found_count_gpu = cuda.mem_alloc(4)
    cuda.memset_d32(found_count_gpu, 0, 1)

    batch_size = 524288  # 512K hashes
    threads_per_block = 256
    blocks = (batch_size * 16 + threads_per_block - 1) // threads_per_block

    print(f"\nBenchmarking JIT kernel with batch size {batch_size:,} ({blocks:,} blocks)...")
    # Warmup
    fn(header_gpu, np.uint64(0), np.uint64(0), dag_gpu, np.uint32(dag_words // 64),
       np.uint32(batch_size), found_nonces_gpu, found_mixes_gpu, found_count_gpu,
       block=(threads_per_block, 1, 1), grid=(blocks, 1))
    cuda.Context.synchronize()

    t0 = time.time()
    iters = 10
    for i in range(iters):
        fn(header_gpu, np.uint64(i * batch_size), np.uint64(0), dag_gpu, np.uint32(dag_words // 64),
           np.uint32(batch_size), found_nonces_gpu, found_mixes_gpu, found_count_gpu,
           block=(threads_per_block, 1, 1), grid=(blocks, 1))
    cuda.Context.synchronize()
    dur = time.time() - t0
    total = batch_size * iters
    mhs = (total / dur) / 1e6
    print(f"\n>>> JIT BENCHMARK HASHRATE: {total:,} hashes in {dur:.2f}s => {mhs:.2f} MH/s! <<<")
