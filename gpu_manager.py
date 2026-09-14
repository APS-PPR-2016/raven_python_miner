#!/usr/bin/env python

# ======================================================================================
# This file acts as a GPU information and management utility.
# It uses the official nvidia-smi library (Python bindings for the NVIDIA
# Management Library) to query and display the status of all available NVIDIA
# GPUs on the machine.
# ======================================================================================

try:
    from nvidia_smi import nvml
except ImportError:
    try:
        import pynvml as nvml
    except ImportError as e:
        print("Error: NVIDIA Management Library (NVML) bindings not installed.")
        print("Please install via: pip install nvidia-ml-py")
        print(e)
        exit()


class Gpu:
    """
    A data class to hold all relevant information for a single GPU device.
    """
    def __init__(self, handle, index):
        self.index = index
        self.handle = handle
        name_raw = nvml.nvmlDeviceGetName(self.handle)
        self.name = name_raw.decode('utf-8') if isinstance(name_raw, bytes) else str(name_raw)

        # Query static and dynamic info at creation time
        self.refresh()

    def refresh(self):
        """Updates the dynamic information for the GPU (utilization, temp, etc.)."""
        mem_info = nvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.mem_total_mb = mem_info.total / 1024**2
        self.mem_used_mb = mem_info.used / 1024**2

        util = nvml.nvmlDeviceGetUtilizationRates(self.handle)
        self.gpu_utilization = util.gpu
        self.mem_utilization = util.memory

        self.temperature = nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)

        try:
            self.power_usage_w = nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except nvml.NVMLError:
            self.power_usage_w = -1 # Mark as not supported


class GpuManager:
    """
    A class that initializes NVML, gathers information on all GPUs,
    and holds it for easy access. Designed to be used as a context manager.
    """
    def __init__(self):
        self.driver_version = None
        self.gpus = []

    def __enter__(self):
        """Initializes NVML and populates GPU info when entering a 'with' block."""
        try:
            nvml.nvmlInit()
            ver_raw = nvml.nvmlSystemGetDriverVersion()
            self.driver_version = ver_raw.decode('utf-8') if isinstance(ver_raw, bytes) else str(ver_raw)
            device_count = nvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                self.gpus.append(Gpu(handle, i))
            return self
        except nvml.NVMLError as e:
            print(f"Error initializing GpuManager: {e}")
            # Ensure shutdown is called even if init fails partially
            nvml.nvmlShutdown()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures NVML is shut down when exiting the 'with' block."""
        nvml.nvmlShutdown()

    def display(self):
        """Prints the collected information in a formatted way."""
        print("--- GPU Manager ---")
        if not self.gpus:
            print("No NVIDIA GPUs found on this machine.")
            return

        print(f"Driver Version: {self.driver_version}\n")
        print(f"Found {len(self.gpus)} NVIDIA GPU(s).\n")

        for gpu in self.gpus:
            print(f"GPU {gpu.index}: {gpu.name}")
            print("-" * 40)
            mem_percent = (gpu.mem_used_mb / gpu.mem_total_mb * 100) if gpu.mem_total_mb > 0 else 0
            print(f"  Memory          : {gpu.mem_used_mb:.2f} MiB / {gpu.mem_total_mb:.2f} MiB ({mem_percent:.1f}%)")
            print(f"  GPU Utilization : {gpu.gpu_utilization} %")
            print(f"  Mem Utilization : {gpu.mem_utilization} %")
            print(f"  Temperature     : {gpu.temperature} C")
            power_str = f"{gpu.power_usage_w:.2f} W" if gpu.power_usage_w != -1 else "Not Supported"
            print(f"  Power Usage     : {power_str}")
            print("")

if __name__ == "__main__":
    # Use the GpuManager as a context manager to ensure cleanup
    try:
        with GpuManager() as manager:
            manager.display()
            # Now you can access the info fingertip, e.g.:
            # if manager.gpus:
            #     print(f"Temperature of first GPU: {manager.gpus[0].temperature} C")
    except nvml.NVMLError:
        print("Could not run GPU Manager. Please ensure NVIDIA drivers are installed.")