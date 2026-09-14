import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from cuda.bindings import nvrtc

cuda_source = r"""
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
#define ROTR32(x, r) (((x) >> (r)) | ((x) << (32 - (r))))
#define FNV_PRIME 0x01000193u
#define FNV1a(h, d) (((h) ^ (d)) * FNV_PRIME)

#define U32TO8_LE(p, v) ((uint8_t*)(p))[0] = (v >>  0) & 0xff; ((uint8_t*)(p))[1] = (v >>  8) & 0xff; \
                         ((uint8_t*)(p))[2] = (v >> 16) & 0xff; ((uint8_t*)(p))[3] = (v >> 24) & 0xff
#define U64TO8_LE(p, v) U32TO8_LE(p, (uint32_t)((v      ) & 0xffffffff)); U32TO8_LE((uint8_t*)(p) + 4, (uint32_t)((v >> 32) & 0xffffffff))

#define ROTR64(x, y) (((x) >> (y)) ^ ((x) << (64 - (y))))

// BLAKE2b constants
static const uint64_t blake2b_iv[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

static const uint64_t sigma[10][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10 },
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0 }
};

__device__ void G(uint64_t *v, int a, int b, int c, int d, uint64_t *m, int m0, int m1, int m2, int m3) {
    v[a] = v[a] + v[b] + m[m0];
    v[d] = ROTR64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = ROTR64(v[b] ^ v[c], 25);
    v[a] = v[a] + v[b] + m[m1];
    v[d] = ROTR64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = ROTR64(v[b] ^ v[c], 11);
    v[a] = v[a] + v[b] + m[m2];
    v[d] = ROTR64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = ROTR64(v[b] ^ v[c], 25);
    v[a] = v[a] + v[b] + m[m3];
    v[d] = ROTR64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = ROTR64(v[b] ^ v[c], 11);
}

__device__ void blake2b_compress(uint64_t m[16], uint64_t v[16], uint64_t h[8], uint64_t f[2]) {
    int i;
    for (i = 0; i < 8; ++i) {
        v[i] = h[i];
        v[i + 8] = blake2b_iv[i];
    }
    v[12] ^= 0x000000000000ffffULL;
    v[13] ^= 0x0000000000000000ULL;
    if (f[0]) {
        v[14] = ~v[14];
    }
    for (i = 0; i < 10; ++i) {
        G(v, 0, 4, 8, 12, m, sigma[i][ 0], sigma[i][ 1], sigma[i][ 2], sigma[i][ 3]);
        G(v, 1, 5, 9, 13, m, sigma[i][ 4], sigma[i][ 5], sigma[i][ 6], sigma[i][ 7]);
        G(v, 2, 6,10,14, m, sigma[i][ 8], sigma[i][ 9], sigma[i][10], sigma[i][11]);
        G(v, 3, 7,11,15, m, sigma[i][12], sigma[i][13], sigma[i][14], sigma[i][15]);
        G(v, 0, 5,10,15, m, sigma[i][ 0], sigma[i][ 5], sigma[i][10], sigma[i][15]);
        G(v, 1, 6,11,12, m, sigma[i][ 4], sigma[i][ 9], sigma[i][14], sigma[i][ 3]);
        G(v, 2, 7, 8,13, m, sigma[i][ 8], sigma[i][13], sigma[i][ 2], sigma[i][ 7]);
        G(v, 3, 4, 9,14, m, sigma[i][12], sigma[i][ 1], sigma[i][ 6], sigma[i][11]);
    }
    for (i = 0; i < 8; ++i) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

__device__ void blake2b_gpu(uint8_t *hash, const uint8_t *input, size_t inlen) {
    uint64_t h[8];
    uint64_t m[16];
    uint64_t f[2] = {0, 0};
    for (int i = 0; i < 8; ++i) h[i] = blake2b_iv[i];

    size_t off = 0;
    while (off < inlen) {
        for (int i = 0; i < 16; ++i) {
            if (off + i*8 < inlen) {
                m[i] = ((uint64_t)input[off + i*8 + 0] << 0) | ((uint64_t)input[off + i*8 + 1] << 8) | ((uint64_t)input[off + i*8 + 2] << 16) | ((uint64_t)input[off + i*8 + 3] << 24) | ((uint64_t)input[off + i*8 + 4] << 32) | ((uint64_t)input[off + i*8 + 5] << 40) | ((uint64_t)input[off + i*8 + 6] << 48) | ((uint64_t)input[off + i*8 + 7] << 56);
            } else {
                m[i] = 0;
            }
        }
        uint64_t v[16];
        blake2b_compress(m, v, h, f);
        off += 128;
    }
    for (int i = 0; i < 4; ++i) {
        U64TO8_LE(hash + i*8, h[i]);
    }
}

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

__device__ uint32_t math_op(uint32_t a, uint32_t b, uint32_t sel) {
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

// CUDA DAG generation kernel: each thread computes one 64-byte half-item
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
        mix[i] = FNV1a(mix[i], gid);
    }

    for (int j = 0; j < 256; ++j) {
        uint32_t t = FNV1a(gid ^ j, mix[j % HASH512_WORDS]);
        uint32_t parent_offset = (t % num_cache_items) * HASH512_WORDS;
        for (int k = 0; k < HASH512_WORDS; ++k) {
            mix[k] = FNV1a(mix[k], light_cache[parent_offset + k]);
        }
    }

    for (int i = 0; i < HASH512_WORDS; ++i) {
        mix[i] = FNV1a(mix[i], gid);
    }

    uint32_t out_offset = gid * HASH512_WORDS;
    for (int i = 0; i < HASH512_WORDS; ++i) {
        dag_out[out_offset + i] = mix[i];
    }
}

// CUDA KAWPOW search kernel accessing the full VRAM DAG dataset
extern "C" __global__ void kawpow_search_dag(const uint8_t *header_pre, uint32_t start_nonce, uint64_t target,
                                             const uint32_t *dag, uint32_t dag_size_items,
                                             uint32_t *found, uint32_t *found_count) {
    uint32_t gid = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = start_nonce + gid;

    uint8_t header[80];
    for (int i = 0; i < 76; i++) header[i] = header_pre[i];
    U32TO8_LE(header + 76, nonce);

    uint8_t hash1[32];
    blake2b_gpu(hash1, header, 80);

    uint32_t mix[8];
    uint32_t seed = *((uint32_t*)hash1);
    for (int i = 0; i < 8; i++) mix[i] = *((uint32_t*)(hash1 + i*4));

    // ProgPOW 64-round random DAG read loop
    kiss99_t rng = {seed, seed + 1, seed + 2, seed + 3};
    uint32_t state[32];
    for (int i = 0; i < 32; i++) state[i] = kiss99(&rng);

    for (int loop = 0; loop < 64; loop++) {
        uint32_t address = kiss99(&rng) % dag_size_items;
        const uint32_t *lookup = dag + (address * 16);
        for (int l = 0; l < 16; l++) {
            state[l] ^= lookup[l];
        }
        for (int i = 0; i < 18; i++) {
            uint32_t src1 = kiss99(&rng) % 16;
            uint32_t src2 = kiss99(&rng) % 16;
            uint32_t sel = kiss99(&rng) % 9;
            uint32_t dst = kiss99(&rng) % 32;
            state[dst] = math_op(state[src1], state[src2], sel);
        }
    }

    for (int i = 0; i < 8; i++) mix[i] = FNV1a(state[i], mix[i]);

    uint8_t final_input[64];
    for (int i = 0; i < 32; i++) final_input[i] = hash1[i];
    for (int i = 0; i < 32; i++) final_input[32 + i] = ((uint8_t*)mix)[i];

    uint8_t final_hash[32];
    blake2b_gpu(final_hash, final_input, 64);

    uint64_t value = ((uint64_t*)final_hash)[0];
    if (value <= target) {
        uint32_t idx = atomicAdd(found_count, 1);
        if (idx < 1024) {
            found[idx] = nonce;
        }
    }
}
"""

print("Compiling CUDA DAG and search kernels with NVRTC...")
dev = pycuda.autoinit.device
cc = dev.compute_capability()
arch = f"sm_{cc[0]}{cc[1]}"

err, prog = nvrtc.nvrtcCreateProgram(cuda_source.encode(), b"kawpow_dag.cu", 0, [], [])
assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
err, = nvrtc.nvrtcCompileProgram(prog, 1, [f"--gpu-architecture={arch}".encode()])
if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
    _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, log)
    print("Compilation error:\n", log.decode())
else:
    _, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * cubin_size
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    mod = cuda.module_from_buffer(cubin)
    print("Compilation successful! Loaded module into CUDA runtime.")
    gen_kernel = mod.get_function("generate_dag_kernel")
    search_kernel = mod.get_function("kawpow_search_dag")
    print(f"generate_dag_kernel max threads: {gen_kernel.max_threads_per_block}, regs: {gen_kernel.num_regs}")
    print(f"kawpow_search_dag max threads: {search_kernel.max_threads_per_block}, regs: {search_kernel.num_regs}")
