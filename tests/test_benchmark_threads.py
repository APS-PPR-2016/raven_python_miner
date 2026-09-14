import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# Load compiled module from main
from main import KawpowRavenMiner

miner = KawpowRavenMiner('rvn.2miners.com:6060', 'RAvjtY1eZxNB3aXe55CQQtKGfezgDXXFuP')
miner.ctx.push()

attrs = miner.device.get_attributes()
sm_count = attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT]
threads = miner.kawpow_search.max_threads_per_block
print(f"Kernel max threads per block: {threads}")

for blocks_per_sm in [16, 32, 64, 128]:
    blocks = sm_count * blocks_per_sm
    batch_size = threads * blocks

    header_gpu = cuda.mem_alloc(76)
    cuda.memcpy_htod(header_gpu, np.zeros(76, dtype=np.uint8))
    found_count_gpu = cuda.mem_alloc(4)
    found_nonce_gpu = cuda.mem_alloc(4 * 1024)
    cuda.memset_d32(found_count_gpu, 0, 1)

    t0 = time.time()
    miner.kawpow_search(header_gpu, np.uint64(0), np.uint64(0x00000000FFFF0000),
                        found_nonce_gpu, found_count_gpu,
                        block=(threads, 1, 1), grid=(blocks, 1))
    cuda.Context.synchronize()
    t1 = time.time()
    dt = t1 - t0

    print(f"Blocks/SM: {blocks_per_sm} | Blocks: {blocks} | Threads: {threads} | Batch: {batch_size:,} nonces | Time: {dt*1000:.1f} ms | Speed: {(batch_size/dt)/1e6:.2f} MH/s")

    header_gpu.free()
    found_count_gpu.free()
    found_nonce_gpu.free()

miner.ctx.pop()
