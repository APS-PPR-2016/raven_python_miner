.\.venv\Scripts\python.exe -c "
import pycuda.autoinit, pycuda.driver as cuda, numpy as np
from cuda.bindings import nvrtc

src = r'''
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
#define ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))

__constant__ uint32_t RC[22] = {
    0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001,
    0x80008081, 0x00008009, 0x0000008A, 0x00000088, 0x80008009, 0x8000000A,
    0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800A, 0x8000000A, 0x80008081, 0x00008080
};

__constant__ uint8_t ROT[5][5] = {
    {0, 4, 3, 9, 18},
    {1, 12, 10, 13, 2},
    {30, 6, 11, 15, 29},
    {28, 23, 25, 21, 24},
    {27, 20, 7, 8, 14}
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
                B[y][(2 * x + 3 * y) % 5] = ROTL32(A[x][y], ROT[x][y]);
            }
        }
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]);
            }
        }
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

extern \"C\" __global__ void test_keccak(uint32_t *st) {
    keccak_f800(st);
}
'''

cc = cuda.Device(0).compute_capability()
arch = f'sm_{cc[0]}{cc[1]}'
err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b'test.cu', 0, [], [])
assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
err, = nvrtc.nvrtcCompileProgram(prog, 1, [f'--gpu-architecture={arch}'.encode()])
_, sz = nvrtc.nvrtcGetCUBINSize(prog)
cubin = b' ' * sz
nvrtc.nvrtcGetCUBIN(prog, cubin)
mod = cuda.module_from_buffer(cubin)
k = mod.get_function('test_keccak')

st = np.zeros(25, dtype=np.uint32)
st_gpu = cuda.mem_alloc(st.nbytes)
cuda.memcpy_htod(st_gpu, st)
k(st_gpu, block=(1, 1, 1), grid=(1, 1))
cuda.memcpy_dtoh(st, st_gpu)
st_gpu.free()
print('GPU Output:', [hex(x) for x in st[:8]])
"