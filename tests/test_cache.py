import hashlib, struct, time, numpy as np

def generate_cache(epoch, cache_size):
    item_count = cache_size // 64
    print(f'[DAG] Generating {item_count:,} light cache items for epoch {epoch}...')
    seed = b'\x00' * 32
    for _ in range(epoch):
        seed = hashlib.sha3_512(seed).digest()

    cache = [hashlib.sha3_512(seed).digest()]
    for _ in range(1, item_count):
        cache.append(hashlib.sha3_512(cache[-1]).digest())

    def fnv1a(v1, v2):
        return ((v1 * 0x01000193) ^ v2) & 0xFFFFFFFF

    for r in range(3):
        t0 = time.time()
        for i in range(item_count):
            prev = struct.unpack('<16I', cache[(i - 1) % item_count])
            first = struct.unpack('<16I', cache[i])
            item = [fnv1a(prev[j], first[j]) for j in range(16)]
            cache[i] = hashlib.sha3_512(struct.pack('<16I', *item)).digest()
        print(f'Round {r+1}/3 finished in {time.time()-t0:.1f}s')

    return np.frombuffer(b''.join(cache), dtype=np.uint32)

t0 = time.time()
c = generate_cache(1, 1000 * 64)
print('Done in', time.time() - t0, 's; shape:', c.shape)