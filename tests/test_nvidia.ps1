.\.venv\Scripts\python.exe -c "import threading, time, pynvml as nvml; from main import KawpowRavenMiner; import pycuda.driver as cuda; nvml.nvmlInit(); handle = nvml.nvmlDeviceGetHandleByIndex(0); miner = KawpowRavenMiner('rvn.2miners.com:6060', 'RAvjtY1eZxNB3aXe55CQQtKGfezgDXXFuP'); miner.ctx.push(); threads, blocks, batch = miner.get_gpu_launch_config(); print(f'Testing {miner.device.name()} with {threads} threads, {blocks} blocks...'); header_gpu = cuda.mem_alloc(76); found_count_gpu = cuda.mem_alloc(4); found_nonce_gpu = cuda.mem_alloc(4096); running = True; 
def monitor():
    while running:
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        print(f'[NVML] GPU Compute Util: {util.gpu}% | Memory Util: {util.memory}% | Power: {power:.1f} W')
        time.sleep(0.5)
t = threading.Thread(target=monitor); t.start()
for _ in range(5):
    cuda.memset_d32(found_count_gpu, 0, 1)
    miner.kawpow_search(header_gpu, cuda.np.uint64(0), cuda.np.uint64(0x00000000FFFF0000), found_nonce_gpu, found_count_gpu, block=(threads, 1, 1), grid=(blocks, 1))
    cuda.Context.synchronize()
running = False; t.join(); miner.ctx.pop(); nvml.nvmlShutdown()"

Testing NVIDIA GeForce RTX 3060 with 768 threads, 3584 blocks...
[NVML] GPU Compute Util: 4% | Memory Util: 0% | Power: 20.7 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 61.5 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 106.2 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 123.2 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 123.4 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 123.6 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 124.1 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 124.3 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 124.4 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 124.7 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 124.8 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 124.9 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 125.0 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 125.5 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 125.4 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 125.5 W
[NVML] GPU Compute Util: 100% | Memory Util: 100% | Power: 125.7 W

