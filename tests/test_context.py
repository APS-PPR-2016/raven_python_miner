import pycuda.driver as cuda
import pycuda.autoinit

cuda.init()
dev = cuda.Device(0)
ctx = dev.make_context()
print("Context created successfully")
ctx.pop()
print("Context popped cleanly")
ctx.detach()
print("Context detached cleanly")
