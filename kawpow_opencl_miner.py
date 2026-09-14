import asyncio
import json
import hashlib
import struct
import time
import numpy as np
import binascii
import random
import math
import os

try:
    import pyopencl as cl
except ImportError:
    print("PyOpenCL not installed. Install with 'pip install pyopencl'")
    raise

class KawPowMiner:
    HASH_BYTES = 32
    EPOCH_LENGTH = 7500  # RVN epoch length for DAG
    CACHE_ROUNDS = 3
    LIGHT_CACHE_ITEM_SIZE = 64
    FULL_DATASET_ITEM_SIZE = 128
    FULL_DATASET_ITEM_PARENTS = 256  # KAWPOW specific

    def __init__(self, pool_url, wallet, worker='grok', gpu_id=0):
        self.pool_host, self.pool_port = pool_url.split(':')
        self.pool_port = int(self.pool_port)
        self.wallet = wallet
        self.worker = worker
        self.extranonce1 = None
        self.extranonce2_size = None
        self.job = None
        self.target = None
        self.height = None
        self.current_epoch = -1
        self.dag_gpu = None
        self.dag_size = 0

        # PyOpenCL setup
        platforms = cl.get_platforms()
        self.platform = platforms[0]  # Use first platform
        devices = self.platform.get_devices()
        self.device = devices[gpu_id]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)

        # OpenCL kernel code for DAG generation and KAWPOW search
        self.opencl_code = """
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define ROTL32(x, r) rotate((x), (uint)(r))
#define ROTR32(x, r) rotate((x), (uint)(32 - (r)))
#define FNV_PRIME 0x01000193u
#define FNV1a(h, d) ((h) ^ (d)) * FNV_PRIME

typedef struct {
    uint z, w, jsr, jcong;
} kiss99_t;

__kernel void generate_dataset_item(__global uint* light_cache, int num_cache_items, uint index, __global uchar* out_item) {
    const int HASH512_WORDS = 16;
    int half_index = get_local_id(0);
    uint seed_val = index * 2 + half_index;

    uint mix[HASH512_WORDS];
    int cache_offset = (seed_val % num_cache_items) * HASH512_WORDS;
    for(int i = 0; i < HASH512_WORDS; ++i) {
        mix[i] = light_cache[cache_offset + i];
    }

    mix[0] ^= seed_val;

    // Simulate Keccak or Blake - for real use Blake2b kernel
    for(int i = 0; i < HASH512_WORDS; ++i) {
        mix[i] = FNV1a(mix[i], seed_val);
    }

    // Main loop for parents
    for (int j = 0; j < 256; ++j) {  // KAWPOW parents
        uint t = FNV1a(seed_val ^ j, mix[j % HASH512_WORDS]);
        int parent_offset = (t % num_cache_items) * HASH512_WORDS;
        for (int k = 0; k < HASH512_WORDS; ++k) {
            mix[k] = FNV1a(mix[k], light_cache[parent_offset + k]);
        }
    }

    // Final mix
    for(int i = 0; i < HASH512_WORDS; ++i) {
        mix[i] = FNV1a(mix[i], seed_val);
    }

    int output_offset = half_index * (HASH512_WORDS * 4);
    __global uint* out_uint = (__global uint*)out_item;
    for (int i = 0; i < HASH512_WORDS; ++i) {
        out_uint[output_offset / 4 + i] = mix[i];
    }
}

uint kiss99(kiss99_t *st) {
    st->z = 36969 * (st->z & 65535) + (st->z >> 16);
    st->w = 18000 * (st->w & 65535) + (st->w >> 16);
    uint mwc = ((st->z << 16) + st->w);
    st->jsr ^= (st->jsr << 17);
    st->jsr ^= (st->jsr >> 13);
    st->jsr ^= (st->jsr << 5);
    st->jcong = 69069 * st->jcong + 1234567;
    return ((mwc ^ st->jcong) + st->jsr);
}

uint math_op(uint a, uint b, uint sel) {
    switch (sel % 9) {
        case 0: return a + b;
        case 1: return a * b;
        case 2: return a >= b ? a - b : b - a;
        case 3: return min(a, b);
        case 4: return max(a, b);
        case 5: return ROTL32(a, b & 31);
        case 6: ROTR32(a, b & 31);
        case 7: return a & b;
        case 8: return a | b;
    }
    return 0;
}

// Blake2b compress (simplified for 80-byte)
void blake2b_compress(ulong h[8], const ulong m[16], bool last) {
    ulong v[16];
    const ulong iv[8] = {0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL, 0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL};
    for (int i = 0; i < 8; i++) v[i] = h[i];
    for (int i = 0; i < 8; i++) v[i + 8] = iv[i];
    v[12] ^= 80; // byte count
    if (last) v[14] = ~v[14];
    // Sigma and G functions - full implementation needed, abbreviated here
    // For real, unroll the 12 rounds with G macros
    // Omit for brevity, assume implemented
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

__kernel void kawpow_search(__global uchar* header_pre, uint start_nonce, ulong target, __global uint* dag, uint dag_size, __global uint* found, __global uint* count) {
    int gid = get_global_id(0);
    uint nonce = start_nonce + gid;
    uchar header[80];
    for (int i = 0; i < 76; i++) header[i] = header_pre[i];
    *((uint*)(header + 76)) = nonce;
    uchar hash1[32];
    ulong h[8] = {0x6a09e667f3bcc908UL ^ 0x01010020UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL, 0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL};
    ulong m[16];
    for (int i = 0; i < 10; i++) m[i] = *((ulong*) (header + i*8));
    for (int i = 10; i < 16; i++) m[i] = 0;
    blake2b_compress(h, m, true);
    for (int i = 0; i < 4; i++) *((ulong*)(hash1 + i*8)) = h[i];
    uint mix[8];
    uint seed = *((uint*)hash1);
    for (int i = 0; i < 8; i++) mix[i] = *((uint*)(hash1 + i*4));
    // ProgPOW loop
    kiss99_t rng = {seed, seed + 1, seed + 2, seed + 3};  // Init RNG
    uint state[32];
    for (int i = 0; i < 32; i++) state[i] = kiss99(&rng);
    for (int loop = 0; loop < 64; loop++) {
        uint address = kiss99(&rng) % dag_size;
        __global uint* lookup = dag + address * 16;
        for (int l = 0; l < 16; l++) {
            state[l] ^= lookup[l];
        }
        for (int i = 0; i < 18; i++) {  // Math
            uint src1 = kiss99(&rng) % 16;
            uint src2 = kiss99(&rng) % 16;
            uint sel = kiss99(&rng) % 9;
            uint dst = kiss99(&rng) % 32;
            state[dst] = math_op(state[src1], state[src2], sel);
        }
    }
    for (int i = 0; i < 8; i++) mix[i] = FNV1a(state[i], mix[i]);
    uchar final_input[64];
    for (int i = 0; i < 32; i++) final_input[i] = hash1[i];
    for (int i = 0; i < 32; i++) final_input[32 + i] = ((uchar*)mix)[i];
    ulong fh[8] = {0x6a09e667f3bcc908UL ^ 0x01010020UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL, 0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL};
    for (int i = 0; i < 8; i++) m[i] = *((ulong*) (final_input + i*8));
    for (int i = 8; i < 16; i++) m[i] = 0;
    blake2b_compress(fh, m, true);
    ulong value = fh[0];
    if (value <= target) {
        uint idx = atomic_inc(count);
        found[idx] = nonce;
    }
}
"""

        self.prg = cl.Program(self.ctx, self.opencl_code).build()

    def __del__(self):
        if self.dag_gpu:
            self.dag_gpu.release()
        self.ctx = None

    def get_epoch(self, block_number):
        return block_number // self.EPOCH_LENGTH

    def get_cache_size(self, epoch):
        size = 16776896 + epoch * 131072 - self.HASH_BYTES
        while not all(size % p == 0 for p in range(2, 18)):
            size -= 2 * self.HASH_BYTES
        return size

    def get_dataset_size(self, epoch):
        size = 1073739904 + epoch * 8388608 - self.HASH_BYTES
        while not all(size % p == 0 for p in range(2, 18)):
            size -= 2 * self.HASH_BYTES
        return size

    def generate_cache(self, epoch):
        seed = b'\x00' * 32
        for _ in range(epoch):
            seed = hashlib.sha3_512(seed).digest()
        cache_size = self.get_cache_size(epoch)
        item_count = cache_size // self.LIGHT_CACHE_ITEM_SIZE
        cache = [hashlib.sha3_512(seed).digest()]
        for i in range(1, item_count):
            cache.append(hashlib.sha3_512(cache[-1]).digest())
        for _ in range(self.CACHE_ROUNDS):
            for i in range(item_count):
                prev = cache[(i - 1) % item_count]
                first = cache[i % item_count]
                item = [fnv1a(prev[j], first[j]) for j in range(self.LIGHT_CACHE_ITEM_SIZE // 4)]
                cache[i] = hashlib.sha3_512(struct.pack('<{}I'.format(self.LIGHT_CACHE_ITEM_SIZE // 4), *item)).digest()
        return np.frombuffer(b''.join(cache), dtype=np.uint32)

    def fnv1a(self, v1, v2):
        return ((v1 * 0x01000193) ^ v2) & 0xFFFFFFFF

    def generate_dag(self, epoch):
        light_cache = self.generate_cache(epoch)
        cache_items = len(light_cache) // (self.LIGHT_CACHE_ITEM_SIZE // 4)
        dataset_size_bytes = self.get_dataset_size(epoch)
        dataset_items = dataset_size_bytes // self.FULL_DATASET_ITEM_SIZE

        light_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=light_cache)
        dag_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, dataset_size_bytes)

        local_work_size = 2
        global_work_size = dataset_items
        self.prg.generate_dataset_item(self.queue, (global_work_size,), (local_work_size,), light_buf, cache_items, dag_buf)
        self.queue.finish()

        return dag_buf, dataset_items

    async def connect_to_pool(self):
        reader, writer = await asyncio.open_connection(self.pool_host, self.pool_port)

        writer.write(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b'\n')
        await writer.drain()
        _ = await reader.readline()
        line = await reader.readline()
        result = json.loads(line)['result']
        self.extranonce1 = binascii.unhexlify(result[1])
        self.extranonce2_size = result[2]

        writer.write(json.dumps({"id": 2, "method": "mining.authorize", "params": [self.wallet, self.worker]}).encode() + b'\n')
        await writer.drain()

        while True:
            line = await reader.readline()
            if not line:
                break
            msg = json.loads(line)
            if 'method' in msg and msg['method'] == 'mining.notify':
                self.job = msg['params']
                self.target = int(msg['params'][2], 16)
                self.height = int(msg['params'][3])
                await self.mine_job(writer)
            elif 'method' in msg and msg['method'] == 'mining.set_difficulty':
                self.target = (1 << 256) // msg['params'][0]

    async def mine_job(self, writer):
        job_id, blob_hex, target_hex, height = self.job[0], self.job[1], self.job[2], self.job[3]
        blob = binascii.unhexlify(blob_hex)

        # Parse blob for coinb1, coinb2, merkle_branch, version, prevhash, ntime, nbits
        # For RVN, blob is the header with placeholders for extranonce2 and nonce
        # Assume standard RVN Stratum blob: 140+ bytes, with ex2 position at coinbase end
        ex2_pos = 42 + len(self.extranonce1)  # Typical position, adjust if needed
        nonce_pos = 108  # Typical for RVN

        extranonce2 = os.urandom(self.extranonce2_size)
        # Insert extranonce2 into blob
        header = blob[:ex2_pos] + extranonce2 + blob[ex2_pos + self.extranonce2_size:nonce_pos]

        epoch = self.get_epoch(self.height)
        if epoch != self.current_epoch:
            if self.dag_gpu:
                self.dag_gpu.release()
            self.dag_gpu, self.dag_size = self.generate_dag(epoch)
            self.current_epoch = epoch

        header_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.frombuffer(header, dtype=np.uint8))
        found_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, 4 * 1024)
        count_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, 4)
        start_nonce = 0
        batch_size = 1000000  # Adjust for GPU
        local_work_size = 256
        global_work_size = batch_size
        while True:
            cl.enqueue_fill_buffer(self.queue, count_buf, np.uint32(0), 0, 4)
            self.prg.kawpow_search(self.queue, (global_work_size,), (local_work_size,), header_buf, np.uint32(start_nonce), np.uint64(self.target), self.dag_gpu, np.uint32(self.dag_size), found_buf, count_buf)
            self.queue.finish()

            count = np.zeros(1, dtype=np.uint32)
            cl.enqueue_copy(self.queue, count, count_buf)
            if count[0] > 0:
                found = np.zeros(1024, dtype=np.uint32)
                cl.enqueue_copy(self.queue, found, found_buf)
                for i in range(count[0]):
                    nonce = found[i]
                    nonce_hex = format(nonce, '08x')
                    ex2_hex = binascii.hexlify(extranonce2).decode()
                    submit = {"id": 4, "method": "mining.submit", "params": [self.wallet, job_id, ex2_hex, nonce_hex]}
                    writer.write(json.dumps(submit).encode() + b'\n')
                    await writer.drain()
                    print(f"Valid hash found! Nonce: {nonce_hex}, Ex2: {ex2_hex}")
            start_nonce += batch_size