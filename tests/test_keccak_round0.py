.\.venv\Scripts\python.exe -c "
import pycuda.autoinit, pycuda.driver as cuda, numpy as np
from cuda.bindings import nvrtc

# Let's inspect test_keccak_cuda vs python step-by-step
with open('test_keccak_cuda.py') as f:
    text = f.read()

# Let's test single round

# Compare Python vs C logic
def rotl32(x, n): return ((x << (n%32)) | (x >> (32 - (n%32)))) & 0xFFFFFFFF
RC = [0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001, 0x80008081, 0x00008009, 0x0000008A, 0x00000088, 0x80008009, 0x8000000A, 0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080, 0x0000800A, 0x8000000A, 0x80008081, 0x00008080]
ROT = [[0, 4, 3, 9, 18], [1, 12, 10, 13, 2], [30, 6, 11, 15, 29], [28, 23, 25, 21, 24], [27, 20, 7, 8, 14]]

def py_keccak(state):
    A = [[state[x + 5*y] for y in range(5)] for x in range(5)]
    for r in range(22):
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x + 4) % 5] ^ rotl32(C[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5): A[x][y] ^= D[x]
        B = [[0]*5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2*x + 3*y) % 5] = rotl32(A[x][y], ROT[x][y])
        for x in range(5):
            for y in range(5):
                A[x][y] = (B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y])) & 0xFFFFFFFF
        A[0][0] ^= RC[r]
    out = [0]*25
    for x in range(5):
        for y in range(5): out[x + 5*y] = A[x][y]
    return out

st0 = [0]*25
py_out = py_keccak(st0)
print('py_out:', [hex(x) for x in py_out[:8]])