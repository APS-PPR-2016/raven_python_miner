import os, time, hashlib, struct, numpy as np
import pycuda.driver as cuda, pycuda.autoinit
from test_cuda_dag_compile import mod

gen_kernel = mod.get_function("generate_dag_kernel")
search_kernel = mod.get_function("kawpow_search_dag")

def get_epoch(height):
    return height // 7500

def get_cache_size(epoch):
    size = 16776896 + epoch * 131072 - 32
    while not all(size % p == 0 for p in range(2, 18)):
        size -= 64
    return size

def get_dataset_size(epoch):
    size = 1073739904 + epoch * 8388608 - 32
    while not all(size % p == 0 for p in range(2, 18)):
        size -= 64
    return size

epoch = get_epoch(4536398)
print(f"Testing epoch {epoch}...")
ds_bytes = get_dataset_size(epoch)
cs_bytes = get_cache_size(epoch)
print(f"Dataset: {ds_bytes/(1024**3):.2f} GB | Cache: {cs_bytes/(1024**2):.2f} MB")

# Quick test with a dummy cache for instant check
item_count = cs_bytes // 64
dummy_cache = np.zeros(item_count * 16, dtype=np.uint32)
cache_gpu = cuda.mem_alloc(dummy_cache.nbytes)
cuda.memcpy_htod(cache_gpu, dummy_cache)

dag_gpu = cuda.mem_alloc(ds_bytes)
total_half_items = ds_bytes // 64

threads = 256
blocks = 28 * 128
print("Running GPU DAG generator...")
t0 = time.time()
gen_kernel(cache_gpu, np.uint32(item_count), dag_gpu, np.uint32(total_half_items),
           block=(threads, 1, 1), grid=(blocks, 1))
cuda.Context.synchronize()
print(f"GPU DAG generator completed in {time.time()-t0:.2f}s!")

cache_gpu.free()
dag_gpu.free()
print("Success!")
