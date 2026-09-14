import time, numpy as np, pycuda.driver as cuda, pycuda.autoinit, pynvml as nvml
from test_cuda_dag_compile import mod
nvml.nvmlInit(); handle = nvml.nvmlDeviceGetHandleByIndex(0)
search_kernel = mod.get_function('kawpow_search_dag')
dag_bytes = 6101615520
dag_items = dag_bytes // 64
dag_gpu = cuda.mem_alloc(dag_bytes)
header_gpu = cuda.mem_alloc(76)
cuda.memcpy_htod(header_gpu, np.zeros(76, dtype=np.uint8))
found_gpu = cuda.mem_alloc(4096)
count_gpu = cuda.mem_alloc(4)

for threads in [256, 512]:
    for blocks_per_sm in [64, 128]:
        blocks = 28 * blocks_per_sm
        batch_size = threads * blocks
        cuda.memset_d32(count_gpu, 0, 1)
        t0 = time.time()
        for _ in range(3):
            search_kernel(header_gpu, np.uint32(0), np.uint64(0x00000000FFFF0000), dag_gpu, np.uint32(dag_items), found_gpu, count_gpu, block=(threads, 1, 1), grid=(blocks, 1))
        cuda.Context.synchronize()
        dt = (time.time() - t0) / 3
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        speed = (batch_size / dt) / 1e6
        print(f'Threads: {threads} | Blocks: {blocks} ({blocks_per_sm}/SM) | Batch: {batch_size:,} | Hashrate: {speed:.2f} MH/s | Mem Util: {util.memory}% | GPU Util: {util.gpu}%')

dag_gpu.free(); header_gpu.free(); found_gpu.free(); count_gpu.free(); nvml.nvmlShutdown()