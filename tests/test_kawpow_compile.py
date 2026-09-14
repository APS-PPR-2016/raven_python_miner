import pycuda.driver as cuda
import pycuda.autoinit
from cuda.bindings import nvrtc

cuda_code = r"""
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

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

__device__ void G(uint64_t *v, int a, int b, int c, int d, const uint64_t *m, uint64_t m0, uint64_t m1, uint64_t m2, uint64_t m3) {
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
    v[12] ^= 0x000000000000ffffULL; // Length low
    v[13] ^= 0x0000000000000000ULL; // Length high
    if (f[0]) {
        v[14] = ~v[14];
    }

    for (i = 0; i < 10; ++i) {
        // Column step
        G(v, 0, 4, 8, 12, m, sigma[i][ 0], sigma[i][ 1], sigma[i][ 2], sigma[i][ 3]);
        G(v, 1, 5, 9, 13, m, sigma[i][ 4], sigma[i][ 5], sigma[i][ 6], sigma[i][ 7]);
        G(v, 2, 6,10,14, m, sigma[i][ 8], sigma[i][ 9], sigma[i][10], sigma[i][11]);
        G(v, 3, 7,11,15, m, sigma[i][12], sigma[i][13], sigma[i][14], sigma[i][15]);
        // Diagonal step
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
        // Load m from input + pad
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
    // Output first 32 bytes
    for (int i = 0; i < 4; ++i) {
        U64TO8_LE(hash + i*8, h[i]);
    }
}

__device__ void progpow_hash_gpu(uint32_t *state, const uint32_t *header_hash) {
    uint32_t matrix[16][16];
    // Initialize matrix from header_hash (32 bytes = 8 uint32)
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            matrix[i][j] = header_hash[(i*16 + j) % 8] ^ (i ^ j);  // Dummy init
        }
    }
    for (int round = 0; round < 64; ++round) {
        uint32_t delta = header_hash[0] ^ round;
        for (int x = 0; x < 16; ++x) {
            for (int y = 0; y < 16; ++y) {
                uint32_t a = matrix[x][y];
                uint32_t b = matrix[(x + 1) % 16][y];
                uint32_t c = delta ^ (x + y * 16);
                int op = (delta >> (x + y)) % 4;
                if (op == 0) a = a + b ^ c;
                else if (op == 1) a = a * b ^ c;
                else if (op == 2) a = a ^ b ^ c;
                else a = (a << 1) | (a >> 31) ^ c;  // Rotate
                matrix[x][y] = a;
            }
        }
        if (round % 4 == 0) {
        }
    }
    // Compress to 32 bytes
    for (int i = 0; i < 8; ++i) {
        state[i] = matrix[0][i] ^ matrix[15][i];
    }
}

// Main mining kernel
extern "C" __global__ void kawpow_search(const uint8_t *header_fixed, uint64_t start_nonce, uint64_t target, uint32_t *found_nonce, uint32_t *found_count) {
    uint64_t nonce = start_nonce + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t full_header[80];
    for (int i = 0; i < 76; ++i) full_header[i] = header_fixed[i];
    U32TO8_LE(full_header + 76, (uint32_t)nonce);

    uint8_t h[32];
    blake2b_gpu(h, full_header, 80);

    uint32_t mix[8];
    progpow_hash_gpu(mix, (uint32_t*)h);

    uint8_t final_input[64];
    for (int i = 0; i < 32; ++i) final_input[i] = h[i];
    for (int i = 0; i < 32; ++i) final_input[32 + i] = ((uint8_t*)mix)[i];
    uint8_t final_hash[32];
    blake2b_gpu(final_hash, final_input, 64);

    uint64_t hash_val = ((uint64_t*)final_hash)[0];
    uint64_t targ_val = target;
    if (hash_val <= targ_val) {
        uint32_t idx = atomicAdd(found_count, 1);
        if (idx < 1024) found_nonce[idx] = (uint32_t)nonce;
    }
}
"""

err, prog = nvrtc.nvrtcCreateProgram(cuda_code.encode(), b"kawpow.cu", 0, [], [])
assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
err, = nvrtc.nvrtcCompileProgram(prog, 1, [b"--gpu-architecture=sm_86"])
_, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
log = b" " * log_size
nvrtc.nvrtcGetProgramLog(prog, log)
print("Compiler Log:\n", log.decode().strip())

if err == nvrtc.nvrtcResult.NVRTC_SUCCESS:
    _, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * cubin_size
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    mod = cuda.module_from_buffer(cubin)
    func = mod.get_function("kawpow_search")
    print("SUCCESS: kawpow_search kernel loaded:", func)
