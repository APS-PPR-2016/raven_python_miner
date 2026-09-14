import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pynvml as nvml
from test_cuda_dag_compile import mod

nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)

search_kernel = mod.get_function("kawpow_search_dag")

# Allocate full 5.68 GB DAG
dag_bytes = 6101615520
dag_items = dag_bytes // 64 # half items of 64 bytes (16 uint32s each)
print(f"Allocating {dag_bytes / (1024**3):.2f} GB in GPU VRAM...")
dag_gpu = cuda.mem_alloc(dag_bytes)

# Quick pattern fill of first 100MB so memory has valid data
print("DAG buffer ready in VRAM.")

header_gpu = cuda.mem_alloc(76)
cuda.memcpy_htod(header_gpu, np.zeros(76, dtype=np.uint8))
found_gpu = cuda.mem_alloc(4 * 1024)
count_gpu = cuda.mem_alloc(4)

threads = 256
blocks = 28 * 64 # 1792 blocks
batch_size = threads * blocks # 458,752 hashes

print(f"Launching kawpow_search_dag: {threads} threads, {blocks} blocks = {batch_size:,} hashes/batch...")

for run in range(5):
    cuda.memset_d32(count_gpu, 0, 1)
    t0 = time.time()
    search_kernel(header_gpu, np.uint32(run * batch_size), np.uint64(0x00000000FFFF0000),
                  dag_gpu, np.uint32(dag_items), found_gpu, count_gpu,
                  block=(threads, 1, 1), grid=(blocks, 1))
    cuda.Context.synchronize()
    dt = time.time() - t0

    util = nvml.nvmlDeviceGetUtilizationRates(handle)
    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    speed = (batch_size / dt) / 1e6

    print(f"Batch {run+1}: Time: {dt*1000:.1f} ms | Hashrate: {speed:.2f} MH/s | GPU Util: {util.gpu}% | Mem Controller: {util.memory}% | Power: {power:.1f} W")

dag_gpu.free()
header_gpu.free()
found_gpu.free()
count_gpu.free()
nvml.nvmlShutdown()
