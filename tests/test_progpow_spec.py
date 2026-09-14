"""
Test official Ravencoin KAWPOW test case 0 against C++ specification logic.
"""

import hashlib
import struct

FNV_PRIME = 0x01000193
FNV_OFFSET_BASIS = 0x811c9dc5

def fnv1(u, v):
    return ((u * FNV_PRIME) ^ v) & 0xFFFFFFFF

def fnv1a(u, v):
    return ((u ^ v) * FNV_PRIME) & 0xFFFFFFFF

def rotl32(x, r):
    r = r & 31
    return (((x << r) | (x >> (32 - r)))) & 0xFFFFFFFF

def rotr32(x, r):
    r = r & 31
    return (((x >> r) | (x << (32 - r)))) & 0xFFFFFFFF

def clz32(x):
    if x == 0: return 32
    return 32 - x.bit_length()

def popcount32(x):
    return bin(x & 0xFFFFFFFF).count('1')

def mul_hi32(a, b):
    return ((a * b) >> 32) & 0xFFFFFFFF

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

    def copy(self):
        return KISS99(self.z, self.w, self.jsr, self.jcong)

RC800 = [
    0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001,
    0x80008081, 0x00008009, 0x0000008A, 0x00000088, 0x80008009, 0x8000000A,
    0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800A, 0x8000000A, 0x80008081, 0x00008080
]

ROT800 = [
    [0, 4, 3, 9, 18],
    [1, 12, 10, 13, 2],
    [30, 6, 11, 15, 29],
    [28, 23, 25, 21, 24],
    [27, 20, 7, 8, 14]
]

def keccak_f800(state):
    A = [[state[x + 5 * y] for y in range(5)] for x in range(5)]
    for r in range(22):
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x + 4) % 5] ^ rotl32(C[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                A[x][y] ^= D[x]
        B = [[0]*5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2*x + 3*y) % 5] = rotl32(A[x][y], ROT800[x][y])
        for x in range(5):
            for y in range(5):
                A[x][y] = B[x][y] ^ ((~B[(x+1)%5][y] & 0xFFFFFFFF) & B[(x+2)%5][y])
        A[0][0] ^= RC800[r]
    for x in range(5):
        for y in range(5):
            state[x + 5 * y] = A[x][y]

RAVENCOIN_KAWPOW = [
    0x72, 0x41, 0x56, 0x45, 0x4E, 0x43, 0x4F, 0x49, 0x4E, 0x4B, 0x41, 0x57, 0x50, 0x4F, 0x57
]

def random_math(a, b, selector):
    s = selector % 11
    if s == 0: return (a + b) & 0xFFFFFFFF
    if s == 1: return (a * b) & 0xFFFFFFFF
    if s == 2: return mul_hi32(a, b)
    if s == 3: return min(a, b)
    if s == 4: return rotl32(a, b)
    if s == 5: return rotr32(a, b)
    if s == 6: return a & b
    if s == 7: return a | b
    if s == 8: return a ^ b
    if s == 9: return (clz32(a) + clz32(b)) & 0xFFFFFFFF
    if s == 10: return (popcount32(a) + popcount32(b)) & 0xFFFFFFFF

def random_merge(a, b, selector):
    x = ((selector >> 16) % 31) + 1
    s = selector % 4
    if s == 0: return ((a * 33) + b) & 0xFFFFFFFF
    if s == 1: return (((a ^ b) * 33)) & 0xFFFFFFFF
    if s == 2: return (rotl32(a, x) ^ b) & 0xFFFFFFFF
    if s == 3: return (rotr32(a, x) ^ b) & 0xFFFFFFFF

# Build light cache for epoch 0
# Epoch 0 seed is 32 zero bytes
def build_epoch0_cache():
    import sha3
    seed = b'\x00' * 32
    # epoch 0 size:
    # 16776896 + 0 - 64 = 16776832
    # In ethash, get_cache_size(0) = 16776896 - 64
    size = 16776896 - 64
    while not all(size % p == 0 for p in range(2, 18)):
        size -= 128
    num_items = size // 64
    print(f"Epoch 0 cache size: {size} bytes, {num_items} items")
    
    # Ethash light cache generation
    cache = bytearray(size)
    curr = hashlib.sha3_512(seed).digest()
    cache[0:64] = curr
    for i in range(1, num_items):
        curr = hashlib.sha3_512(curr).digest()
        cache[i*64:(i+1)*64] = curr
    
    for _ in range(3):
        for i in range(num_items):
            v = struct.unpack('<I', cache[i*64:i*64+4])[0] % num_items
            item_curr = cache[i*64:(i+1)*64]
            item_ref = cache[v*64:(v+1)*64]
            xor_item = bytes(a ^ b for a, b in zip(item_curr, item_ref))
            cache[i*64:(i+1)*64] = hashlib.sha3_512(xor_item).digest()
            
    return bytes(cache), num_items

if __name__ == '__main__':
    print("Testing test case 0 setup...")
