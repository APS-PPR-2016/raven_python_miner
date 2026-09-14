
import hashlib, struct, time, os, numpy as np

def get_cache_size(epoch):
    size = 16776896 + epoch * 131072 - 32
    while not all(size % p == 0 for p in range(2, 18)):
        size -= 64
    return size

epoch = 604
cache_size = get_cache_size(epoch)
item_count = cache_size // 64
cache_file = f'.cache_rvn_epoch_{epoch}.bin'

print(f'Generating light cache ({item_count:,} items)...')
t0 = time.time()
seed = b'\x00' * 32
for _ in range(epoch):
    seed = hashlib.sha3_512(seed).digest()

cache = [hashlib.sha3_512(seed).digest()]
for _ in range(1, item_count):
    cache.append(hashlib.sha3_512(cache[-1]).digest())

def fnv1a(v1, v2):
    return ((v1 * 0x01000193) ^ v2) & 0xFFFFFFFF

for r in range(3):
    t_r = time.time()
    for i in range(item_count):
        prev = struct.unpack('<16I', cache[(i - 1) % item_count])
        first = struct.unpack('<16I', cache[i])
        item = [fnv1a(prev[j], first[j]) for j in range(16)]
        cache[i] = hashlib.sha3_512(struct.pack('<16I', *item)).digest()
    print(f'Round {r+1}/3 finished in {time.time()-t_r:.1f}s')

arr = np.frombuffer(b''.join(cache), dtype=np.uint32)
arr.tofile(cache_file)
print(f'Generated and saved {cache_file} in {time.time()-t0:.1f}s ({os.path.getsize(cache_file)/1024**2:.1f} MB)')
