import struct
import hashlib

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

class MixRngState:
    def __init__(self, seed):
        seed_lo = seed[0]
        seed_hi = seed[1]
        z = fnv1a(FNV_OFFSET_BASIS, seed_lo)
        w = fnv1a(z, seed_hi)
        jsr = fnv1a(w, seed_lo)
        jcong = fnv1a(jsr, seed_hi)
        self.rng = KISS99(z, w, jsr, jcong)
        self.dst_seq = list(range(32))
        self.src_seq = list(range(32))
        for i in range(32, 1, -1):
            r1 = self.rng() % i
            self.dst_seq[i - 1], self.dst_seq[r1] = self.dst_seq[r1], self.dst_seq[i - 1]
            r2 = self.rng() % i
            self.src_seq[i - 1], self.src_seq[r2] = self.src_seq[r2], self.src_seq[i - 1]
        self.dst_counter = 0
        self.src_counter = 0

    def copy(self):
        c = MixRngState.__new__(MixRngState)
        c.rng = self.rng.copy()
        c.dst_seq = list(self.dst_seq)
        c.src_seq = list(self.src_seq)
        c.dst_counter = self.dst_counter
        c.src_counter = self.src_counter
        return c

    def next_dst(self):
        v = self.dst_seq[self.dst_counter % 32]
        self.dst_counter += 1
        return v

    def next_src(self):
        v = self.src_seq[self.src_counter % 32]
        self.src_counter += 1
        return v

def build_light_cache(num_items, seed=b'\x00'*32):
    cache = bytearray(num_items * 64)
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
    return cache

def is_odd_prime(n):
    d = 3
    while d * d <= n:
        if n % d == 0: return False
        d += 2
    return True

def find_largest_prime(n):
    if n < 2: return 0
    if n == 2: return 2
    if n % 2 == 0: n -= 1
    while not is_odd_prime(n):
        n -= 2
    return n

def calc_dataset_item_512(cache_bytes, num_cache_items, index):
    offset = (index % num_cache_items) * 64
    mix = bytearray(cache_bytes[offset:offset+64])
    w0 = struct.unpack('<I', mix[0:4])[0] ^ (index & 0xFFFFFFFF)
    mix[0:4] = struct.pack('<I', w0)
    mix = bytearray(hashlib.sha3_512(mix).digest())
    
    # 512 update rounds
    mix_words = list(struct.unpack('<16I', mix))
    for j in range(512):
        t = fnv1((index ^ j) & 0xFFFFFFFF, mix_words[j % 16])
        parent_idx = t % num_cache_items
        parent_words = struct.unpack('<16I', cache_bytes[parent_idx*64:(parent_idx+1)*64])
        for k in range(16):
            mix_words[k] = fnv1(mix_words[k], parent_words[k])
    
    mix_packed = struct.pack('<16I', *mix_words)
    return hashlib.sha3_512(mix_packed).digest()

def calc_dataset_item_2048(cache_bytes, num_cache_items, index):
    h0 = calc_dataset_item_512(cache_bytes, num_cache_items, index * 4 + 0)
    h1 = calc_dataset_item_512(cache_bytes, num_cache_items, index * 4 + 1)
    h2 = calc_dataset_item_512(cache_bytes, num_cache_items, index * 4 + 2)
    h3 = calc_dataset_item_512(cache_bytes, num_cache_items, index * 4 + 3)
    return struct.unpack('<64I', h0 + h1 + h2 + h3)

print("Setup completed. Building epoch 0 cache (this takes ~1-2s)...")
num_cache_items = find_largest_prime((1 << 24) // 64)
print(f"Num cache items: {num_cache_items}")
cache = build_light_cache(num_cache_items)
print("Cache built!")

# Build L1 cache (first 16384 bytes of DAG = 64 items of hash2048 = 4096 uint32s)
print("Building L1 cache (64 items)...")
l1_words = []
for i in range(64):
    l1_words.extend(calc_dataset_item_2048(cache, num_cache_items, i))
print(f"L1 cache built: {len(l1_words)} words")

def run_test(reset_state_each_round=True):
    # Test Case 0: block=0, header=0, nonce=0
    header_words = [0]*8
    nonce = 0
    st = [0]*25
    for i in range(8): st[i] = header_words[i]
    st[8] = nonce & 0xFFFFFFFF
    st[9] = nonce >> 32
    for i in range(10, 25): st[i] = RAVENCOIN_KAWPOW[i - 10]
    keccak_f800(st)
    state2 = st[:8]

    # init mix
    z = fnv1a(FNV_OFFSET_BASIS, state2[0])
    w = fnv1a(z, state2[1])
    mix = [[0]*32 for _ in range(16)]
    for l in range(16):
        jsr = fnv1a(w, l)
        jcong = fnv1a(jsr, l)
        rng = KISS99(z, w, jsr, jcong)
        for r in range(32):
            mix[l][r] = rng()

    period = 0
    init_st = [period & 0xFFFFFFFF, period >> 32]
    master_state = MixRngState(init_st)

    num_items = find_largest_prime((1 << 30) // 128) // 2

    dag_cache = {}

    for r in range(64):
        if reset_state_each_round:
            st_round = master_state.copy()
        else:
            st_round = master_state

        item_index = mix[r % 16][0] % num_items
        if item_index not in dag_cache:
            dag_cache[item_index] = calc_dataset_item_2048(cache, num_cache_items, item_index)
        dag_item = dag_cache[item_index]

        # max operations = 18
        for i in range(18):
            if i < 11:
                src = st_round.next_src()
                dst = st_round.next_dst()
                sel = st_round.rng()
                for l in range(16):
                    offset = mix[l][src] % 4096
                    mix[l][dst] = random_merge(mix[l][dst], l1_words[offset], sel)
            if i < 18:
                src_rnd = st_round.rng() % (32 * 31)
                src1 = src_rnd % 32
                src2 = src_rnd // 32
                if src2 >= src1: src2 += 1
                sel1 = st_round.rng()
                dst = st_round.next_dst()
                sel2 = st_round.rng()
                for l in range(16):
                    data = random_math(mix[l][src1], mix[l][src2], sel1)
                    mix[l][dst] = random_merge(mix[l][dst], data, sel2)

        dsts = [0]*4
        sels = [0]*4
        for i in range(4):
            dsts[i] = 0 if i == 0 else st_round.next_dst()
            sels[i] = st_round.rng()

        for l in range(16):
            offset = ((l ^ r) % 16) * 4
            for i in range(4):
                word = dag_item[offset + i]
                mix[l][dsts[i]] = random_merge(mix[l][dsts[i]], word, sels[i])

    # Reduce mix
    lane_hash = [0]*16
    for l in range(16):
        lh = FNV_OFFSET_BASIS
        for i in range(32):
            lh = fnv1a(lh, mix[l][i])
        lane_hash[l] = lh

    mix_hash = [FNV_OFFSET_BASIS]*8
    for l in range(16):
        mix_hash[l % 8] = fnv1a(mix_hash[l % 8], lane_hash[l])

    mix_bytes = struct.pack('<8I', *mix_hash)
    return mix_bytes.hex()

expected_mix = "6e97b47b134fda0c7888802988e1a373affeb28bcd813b6e9a0fc669c935d03a"
print("Running with reset_state_each_round=True...")
res_reset = run_test(reset_state_each_round=True)
print(f"Result (reset):    {res_reset}")
print(f"Expected:          {expected_mix}")
print(f"Match (reset):     {res_reset == expected_mix}")

if res_reset != expected_mix:
    print("\nRunning with reset_state_each_round=False...")
    res_no_reset = run_test(reset_state_each_round=False)
    print(f"Result (no reset): {res_no_reset}")
    print(f"Expected:          {expected_mix}")
    print(f"Match (no reset):  {res_no_reset == expected_mix}")
