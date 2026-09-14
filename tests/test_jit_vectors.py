import pycuda.driver as cuda
import pycuda.autoinit
from cuda.bindings import nvrtc
import numpy as np
import time

import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('tests'))

from test_jit_progpow import compile_jit

# Official test cases
test_cases = [
    {
        "block": 0,
        "period": 0,
        "header": "0000000000000000000000000000000000000000000000000000000000000000",
        "nonce": 0x0,
        "exp_mix": "6e97b47b134fda0c7888802988e1a373affeb28bcd813b6e9a0fc669c935d03a",
        "exp_final": "e601a7257a70dc48fccc97a7330d704d776047623b92883d77111fb36870f3d1"
    },
    {
        "block": 49,
        "period": 16,
        "header": "63155f732f2bf556967f906155b510c917e48e99685ead76ea83f4eca03ab12b",
        "nonce": 0x7073c07,
        "exp_mix": "d36f7e815ee09e74eceb9c96993a3d681edf2bf0921fc7bb710364042db99777",
        "exp_final": "e7ced124598fd2500a55ad9f9f48e3569327fe50493c77a4ac9799b96efb9463"
    },
    {
        "block": 50,
        "period": 16,
        "header": "9e7248f20914913a73d80a70174c331b1d34f260535ac3631d770e656b5dd922",
        "nonce": 0x76e482e,
        "exp_mix": "d6dc634ae837e2785b347648ea515e25e5d8821ae0b95e1c2a9c2d497e0dcfbd",
        "exp_final": "ab0ad7ef8d8ee317dd12d10310aceed7321d34fb263791c2de5776a6658d177e"
    }
]

from kawpow_full_engine import KawpowGpuEngine

engine = KawpowGpuEngine(0)
engine.init_epoch(0)

print("\n--- Verifying JIT Kernel against Official Test Vectors ---")
jit_kernels = {}

for tc in test_cases:
    period = tc["period"]
    if period not in jit_kernels:
        print(f"JIT compiling period {period}...")
        jit_kernels[period] = compile_jit(period)
    fn = jit_kernels[period]

    header_bytes = bytes.fromhex(tc["header"])
    header_words = np.frombuffer(header_bytes, dtype=np.uint32)
    header_gpu = cuda.mem_alloc(32)
    cuda.memcpy_htod(header_gpu, header_words)

    max_found = 1024
    found_nonces_gpu = cuda.mem_alloc(max_found * 8)
    found_mixes_gpu = cuda.mem_alloc(max_found * 32)
    found_count_gpu = cuda.mem_alloc(4)
    cuda.memset_d32(found_count_gpu, 0, 1)

    # 1 hash = 16 threads
    fn(header_gpu, np.uint64(tc["nonce"]), np.uint64(0xFFFFFFFFFFFFFFFF),
       engine.dag_gpu, np.uint32(engine.num_dag_items_2048),
       np.uint32(1), found_nonces_gpu, found_mixes_gpu, found_count_gpu,
       block=(16, 1, 1), grid=(1, 1))
    cuda.Context.synchronize()

    cnt = np.zeros(1, dtype=np.uint32)
    cuda.memcpy_dtoh(cnt, found_count_gpu)
    assert cnt[0] > 0

    mixes = np.zeros(8, dtype=np.uint32)
    cuda.memcpy_dtoh(mixes, found_mixes_gpu)
    mix_hex = bytes(mixes).hex()

    print(f"Block {tc['block']} (period {period}): Mix={mix_hex} | Match={mix_hex == tc['exp_mix']}")
    assert mix_hex == tc['exp_mix']

print("\nALL JIT TEST VECTORS PASSED WITH 100% BITWISE PARITY!")
