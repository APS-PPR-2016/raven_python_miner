import pycuda.driver as cuda
import pycuda.autoinit
from cuda.bindings import nvrtc
import numpy as np
import time

# Let's test placing the 16KB L1 cache in shared memory
CUDA_SHARED_TEST = r"""
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned char uint8_t;

#define PROGPOW_LANES 16
#define PROGPOW_REGS 32
#define PROGPOW_CACHE_WORDS 4096

#define ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
#define ROTR32(x, r) (((x) >> (r)) | ((x) << (32 - (r))))
#define FNV_PRIME 0x01000193u
#define FNV_OFFSET_BASIS 0x811c9dc5u

__device__ __forceinline__ uint32_t fnv1a(uint32_t u, uint32_t v) {
    return (u ^ v) * FNV_PRIME;
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

__device__ __forceinline__ uint32_t random_math(uint32_t a, uint32_t b, uint32_t selector) {
    switch (selector % 11) {
        default:
        case 0: return a + b;
        case 1: return a * b;
        case 2: return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
        case 3: return (a < b) ? a : b;
        case 4: return ROTL32(a, b & 31);
        case 5: return ROTR32(a, b & 31);
        case 6: return a & b;
        case 7: return a | b;
        case 8: return a ^ b;
        case 9: return __clz(a) + __clz(b);
        case 10: return __popc(a) + __popc(b);
    }
}

__device__ __forceinline__ void random_merge(uint32_t &a, uint32_t b, uint32_t selector) {
    uint32_t x = ((selector >> 16) % 31) + 1;
    switch (selector % 4) {
        case 0: a = (a * 33) + b; break;
        case 1: a = (a ^ b) * 33; break;
        case 2: a = ROTL32(a, x) ^ b; break;
        case 3: a = ROTR32(a, x) ^ b; break;
    }
}

extern "C" __global__ void benchmark_shared(const uint32_t *dag, uint32_t num_dag_items,
                                            uint32_t batch_size, uint64_t *out) {
    __shared__ uint32_t shared_l1[PROGPOW_CACHE_WORDS];

    // Cooperatively load 4096 words (16KB) into shared memory
    for (int idx = threadIdx.x; idx < PROGPOW_CACHE_WORDS; idx += blockDim.x) {
        shared_l1[idx] = dag[idx];
    }
    __syncthreads();

    uint32_t lane_id = threadIdx.x % 16;
    uint32_t group_id = ((blockIdx.x * blockDim.x) + threadIdx.x) / 16;
    if (group_id >= batch_size) return;

    uint32_t mix[32];
    for (int i = 0; i < 32; i++) mix[i] = group_id + i + lane_id;

    kiss99_t prog_rng_init = {12345, 67890, 54321, 98765};
    uint8_t dst_seq[32];
    uint8_t src_seq[32];
    for (int i = 0; i < 32; i++) { dst_seq[i] = i; src_seq[i] = i; }

    for (int r = 0; r < 64; r++) {
        uint32_t mix_r0 = __shfl_sync(0xffff, mix[0], r % 16);
        uint32_t item_index = mix_r0 % num_dag_items;

        kiss99_t prog_rng = prog_rng_init;
        int dst_counter = 0;
        int src_counter = 0;

        #pragma unroll
        for (int i = 0; i < 18; i++) {
            if (i < 11) {
                uint32_t src = src_seq[(src_counter++) % 32];
                uint32_t dst = dst_seq[(dst_counter++) % 32];
                uint32_t sel = kiss99(&prog_rng);
                uint32_t offset = mix[src] % PROGPOW_CACHE_WORDS;
                // Read from shared memory!
                random_merge(mix[dst], shared_l1[offset], sel);
            }
            if (i < 18) {
                uint32_t src_rnd = kiss99(&prog_rng) % (32 * 31);
                uint32_t src1 = src_rnd % 32;
                uint32_t src2 = src_rnd / 32;
                if (src2 >= src1) src2++;

                uint32_t sel1 = kiss99(&prog_rng);
                uint32_t dst = dst_seq[(dst_counter++) % 32];
                uint32_t sel2 = kiss99(&prog_rng);

                uint32_t data = random_math(mix[src1], mix[src2], sel1);
                random_merge(mix[dst], data, sel2);
            }
        }

        uint32_t dsts[4];
        uint32_t sels[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            dsts[i] = (i == 0) ? 0 : dst_seq[(dst_counter++) % 32];
            sels[i] = kiss99(&prog_rng);
        }

        uint32_t dag_word_offset = (item_index * 64) + (((lane_id ^ r) % 16) * 4);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // Read from global memory DAG
            uint32_t word = __ldg(&dag[dag_word_offset + i]);
            random_merge(mix[dsts[i]], word, sels[i]);
        }
    }

    if (lane_id == 0 && group_id == 0) {
        *out = mix[0];
    }
}
"""

def test():
    dev = cuda.Device(0)
    cc = dev.compute_capability()
    arch = f"sm_{cc[0]}{cc[1]}"
    err, prog = nvrtc.nvrtcCreateProgram(CUDA_SHARED_TEST.encode(), b"bench.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
    err, = nvrtc.nvrtcCompileProgram(prog, 1, [f"--gpu-architecture={arch}".encode()])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        print("Compile failed:", log.decode())
        return

    _, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * cubin_size
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    mod = cuda.module_from_buffer(cubin)
    fn = mod.get_function("benchmark_shared")

    # Allocate dummy DAG (100 MB)
    dag_words = 25 * 1024 * 1024
    dag_gpu = cuda.mem_alloc(dag_words * 4)
    cuda.memset_d32(dag_gpu, 0x12345678, dag_words)
    out_gpu = cuda.mem_alloc(8)

    batch_size = 524288  # 512K hashes
    threads_per_block = 256
    blocks = (batch_size * 16 + threads_per_block - 1) // threads_per_block

    print(f"Launching {batch_size:,} hashes with shared memory L1 cache...")
    # Warmup
    fn(dag_gpu, np.uint32(dag_words // 64), np.uint32(batch_size), out_gpu,
       block=(threads_per_block, 1, 1), grid=(blocks, 1))
    cuda.Context.synchronize()

    t0 = time.time()
    iters = 10
    for _ in range(iters):
        fn(dag_gpu, np.uint32(dag_words // 64), np.uint32(batch_size), out_gpu,
           block=(threads_per_block, 1, 1), grid=(blocks, 1))
    cuda.Context.synchronize()
    dur = time.time() - t0
    total = batch_size * iters
    mhs = (total / dur) / 1e6
    print(f"Speed: {total:,} hashes in {dur:.2f}s => {mhs:.2f} MH/s!")

if __name__ == '__main__':
    test()
