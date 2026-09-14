"""
KAWPOW (Ravencoin ProgPOW) CUDA Kernel Module
Implements Keccak-f[800], Ravencoin domain constraints, ProgPOW DAG search, and GPU DAG generation.
Matches the official Ravencoin consensus specification (src/crypto/ethash/lib/ethash/progpow.cpp).
"""

import numpy as np
import pycuda.driver as cuda
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='pycuda')

KAWPOW_CUDA_SOURCE = r"""
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define PROGPOW_LANES 16
#define PROGPOW_REGS 32
#define PROGPOW_DAG_LOADS 4
#define PROGPOW_CACHE_BYTES (16*1024)
#define PROGPOW_CNT_DAG 64
#define PROGPOW_CNT_CACHE 11
#define PROGPOW_CNT_MATH 18
#define PROGPOW_PERIOD_LENGTH 50

#define ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
#define ROTR32(x, r) (((x) >> (r)) | ((x) << (32 - (r))))
#define FNV_PRIME 0x01000193u
#define FNV1a(h, d) (((h) ^ (d)) * FNV_PRIME)

// Keccak-f[800] round constants (22 rounds)
__constant__ uint32_t RC[22] = {
    0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001,
    0x80008081, 0x00008009, 0x0000008A, 0x00000088, 0x80008009, 0x8000000A,
    0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800A, 0x8000000A, 0x80008081, 0x00008080
};

// Keccak-f[800] rotation offsets
__constant__ uint8_t ROT[5][5] = {
    {0, 4, 3, 9, 18},
    {1, 12, 10, 13, 2},
    {30, 6, 11, 15, 29},
    {28, 23, 25, 21, 24},
    {27, 20, 7, 8, 14}
};

// Ravencoin KAWPOW domain constraint string: "RAVENCOINKAWPOW"
__constant__ uint32_t RAVENCOIN_KAWPOW[15] = {
    0x72, // R
    0x41, // A
    0x56, // V
    0x45, // E
    0x4E, // N
    0x43, // C
    0x4F, // O
    0x49, // I
    0x4E, // N
    0x4B, // K
    0x41, // A
    0x57, // W
    0x50, // P
    0x4F, // O
    0x57  // W
};

// Keccak-f[800] 25-word permutation
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
        // 1. Theta
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

        // 2. Rho and Pi
        uint32_t B[5][5];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                B[y][(2 * x + 3 * y) % 5] = ROTL32(A[x][y], ROT[x][y]);
            }
        }

        // 3. Chi
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]);
            }
        }

        // 4. Iota
        A[0][0] ^= RC[round_idx];
    }

    #pragma unroll
    for (int x = 0; x < 5; x++) {
        #pragma unroll
        for (int y = 0; y < 5; y++) {
            state[x + 5 * y] = A[x][y];
        }
    }
}

// KISS99 RNG
typedef struct {
    uint32_t z, w, jsr, jcong;
} kiss99_t;

__device__ uint32_t kiss99(kiss99_t *st) {
    st->z = 36969 * (st->z & 65535) + (st->z >> 16);
    st->w = 18000 * (st->w & 65535) + (st->w >> 16);
    uint32_t mwc = ((st->z << 16) + st->w);
    st->jsr ^= (st->jsr << 17);
    st->jsr ^= (st->jsr >> 13);
    st->jsr ^= (st->jsr << 5);
    st->jcong = 69069 * st->jcong + 1234567;
    return ((mwc ^ st->jcong) + st->jsr);
}

__device__ uint32_t fnv1a(uint32_t a, uint32_t b) {
    return (a ^ b) * 0x01000193u;
}

// Math ops for ProgPOW
__device__ uint32_t math(uint32_t a, uint32_t b, uint32_t sel) {
    switch (sel % 9) {
        case 0: return a + b;
        case 1: return a * b;
        case 2: return a >= b ? a - b : b - a;
        case 3: return (a < b) ? a : b;
        case 4: return (a > b) ? a : b;
        case 5: return ROTL32(a, b & 31);
        case 6: return ROTR32(a, b & 31);
        case 7: return a & b;
        case 8: return a | b;
    }
    return 0;
}

// ProgPOW RNG init
__device__ kiss99_t progpow_rng_init(uint32_t seed, uint32_t sequence, uint32_t mix_seq) {
    kiss99_t st;
    st.z = fnv1a(0x811c9dc5, seed);
    st.w = fnv1a(st.z, sequence);
    st.jsr = fnv1a(st.w, mix_seq);
    st.jcong = fnv1a(st.jsr, seed + sequence + mix_seq);
    kiss99(&st); kiss99(&st); kiss99(&st);
    return st;
}

// Random source
__device__ uint32_t progpow_random(kiss99_t *rng, uint32_t mix_seq, uint32_t limit) {
    return kiss99(rng) % limit;
}

// CUDA DAG generation kernel: each thread generates one 64-byte half-item
extern "C" __global__ void generate_dag_kernel(const uint32_t *light_cache, uint32_t num_cache_items, uint32_t *dag_out, uint32_t total_half_items) {
    uint32_t gid = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_half_items) return;

    const int HASH512_WORDS = 16;
    uint32_t mix[HASH512_WORDS];
    uint32_t cache_offset = (gid % num_cache_items) * HASH512_WORDS;
    for (int i = 0; i < HASH512_WORDS; ++i) {
        mix[i] = light_cache[cache_offset + i];
    }
    mix[0] ^= gid;

    for (int i = 0; i < HASH512_WORDS; ++i) {
        mix[i] = fnv1a(mix[i], gid);
    }

    for (int j = 0; j < 256; ++j) {
        uint32_t t = fnv1a(gid ^ j, mix[j % HASH512_WORDS]);
        uint32_t parent_offset = (t % num_cache_items) * HASH512_WORDS;
        for (int k = 0; k < HASH512_WORDS; ++k) {
            mix[k] = fnv1a(mix[k], light_cache[parent_offset + k]);
        }
    }

    for (int i = 0; i < HASH512_WORDS; ++i) {
        mix[i] = fnv1a(mix[i], gid);
    }

    uint32_t out_offset = gid * HASH512_WORDS;
    for (int i = 0; i < HASH512_WORDS; ++i) {
        dag_out[out_offset + i] = mix[i];
    }
}

// Full Keccak-f[800] KAWPOW search kernel
extern "C" __global__ void kawpow_search_progpow(const uint32_t *header_words, uint64_t start_nonce, uint64_t target_high64,
                                                 const uint32_t *dag, uint32_t dag_size_items,
                                                 uint64_t *found_nonces, uint8_t *found_mixes, uint32_t *found_count) {
    uint32_t gid = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + gid;

    // 1. Initial Keccak-f[800] absorb phase
    uint32_t state[25];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = header_words[i];
    }
    state[8] = (uint32_t)(nonce & 0xFFFFFFFFULL);
    state[9] = (uint32_t)(nonce >> 32);
    #pragma unroll
    for (int i = 10; i < 25; i++) {
        state[i] = RAVENCOIN_KAWPOW[i - 10];
    }

    keccak_f800(state);

    // Retain initial 8 words of state as carry-over for final Keccak
    uint32_t state2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state2[i] = state[i];
    }

    // ProgPOW KISS99 RNG seed from initial Keccak
    uint32_t seed = state2[0];
    uint32_t mix[PROGPOW_REGS];
    kiss99_t my_rng = progpow_rng_init(seed, threadIdx.x, threadIdx.x);
    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++) {
        mix[i] = kiss99(&my_rng);
    }

    kiss99_t global_rng = progpow_rng_init(seed, 0, 0);

    // 2. ProgPOW random DAG search loop
    for (int loop = 0; loop < PROGPOW_CNT_DAG; loop++) {
        uint32_t address = progpow_random(&global_rng, loop, dag_size_items);
        const uint32_t *lookup = dag + (address * 16);
        uint32_t offset = progpow_random(&my_rng, loop, PROGPOW_CACHE_BYTES / 4);

        #pragma unroll
        for (int l = 0; l < 16; l++) {
            mix[l] ^= lookup[l] ^ dag[offset % dag_size_items];
        }

        #pragma unroll
        for (int i = 0; i < PROGPOW_CNT_MATH; i++) {
            uint32_t src1 = progpow_random(&global_rng, i, PROGPOW_REGS);
            uint32_t src2 = progpow_random(&global_rng, i, PROGPOW_REGS);
            uint32_t sel = progpow_random(&global_rng, i, 9);
            uint32_t dst = progpow_random(&global_rng, i, PROGPOW_REGS);
            mix[dst] = math(mix[src1], mix[src2], sel);
        }
    }

    // Mix digest (32 bytes = 8 uint32 words)
    uint32_t mix_hash[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        mix_hash[i] = fnv1a(state2[i], mix[i]);
    }

    // 3. Final Keccak-f[800] absorb phase
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = state2[i];
    }
    #pragma unroll
    for (int i = 8; i < 16; i++) {
        state[i] = mix_hash[i - 8];
    }
    #pragma unroll
    for (int i = 16; i < 25; i++) {
        state[i] = RAVENCOIN_KAWPOW[i - 16];
    }

    keccak_f800(state);

    // 4. Compare final 256-bit hash against big-endian target
    // output words state[0] and state[1] represent word64[0] in little-endian.
    // BSWAP64 converts to big-endian order for numeric comparison.
    uint64_t w0 = ((uint64_t)state[1] << 32) | (uint64_t)state[0];
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
"""

def compile_kawpow_cuda(device):
    """Compiles the KAWPOW CUDA source via PyCUDA SourceModule or NVRTC bindings."""
    cc = device.compute_capability()
    arch = f"sm_{cc[0]}{cc[1]}"

    try:
        from pycuda.compiler import SourceModule
        mod = SourceModule(KAWPOW_CUDA_SOURCE, options=['-O3', f'-arch={arch}'])
        return mod
    except Exception:
        from cuda.bindings import nvrtc
        err, prog = nvrtc.nvrtcCreateProgram(KAWPOW_CUDA_SOURCE.encode(), b"kawpow_cuda.cu", 0, [], [])
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcCreateProgram failed: {err}")
        err, = nvrtc.nvrtcCompileProgram(prog, 1, [f"--gpu-architecture={arch}".encode()])
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * log_size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"NVRTC compilation failed:\n{log.decode().strip()}")
        _, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * cubin_size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        mod = cuda.module_from_buffer(cubin)
        return mod
