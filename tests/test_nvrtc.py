import pycuda.driver as cuda
import pycuda.autoinit
from cuda.bindings import nvrtc

code = """
extern "C" __global__ void add(int *a, int b) {
    *a += b;
}
"""

err, prog = nvrtc.nvrtcCreateProgram(code.encode(), b"add.cu", 0, [], [])
assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS
# Compile directly to sm_86 CUBIN
err, = nvrtc.nvrtcCompileProgram(prog, 1, [b"--gpu-architecture=sm_86"])
if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
    _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, log)
    print("Compilation error:", log.decode())
else:
    _, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * cubin_size
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    mod = cuda.module_from_buffer(cubin)
    func = mod.get_function("add")
    print("SUCCESS: CUBIN loaded directly via NVRTC & PyCUDA:", func)
