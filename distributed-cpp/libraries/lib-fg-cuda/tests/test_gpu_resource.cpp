#include "fgp/cuda/cuda_nvml.h"
#include "fgp/cuda/gpu_resource.h"

#include <iostream>
#include <iomanip>

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "         FGP-CUDA GPU Resource Test Project       " << std::endl;
    std::cout << "==================================================" << std::endl;

    // 1. Initialize NVML using CudaNVML
    fgp::cuda::CudaNVML nvml;
    nvmlReturn_t status = nvml.initialize();
    if (status != NVML_SUCCESS) {
        const char* err = nvml.getErrorString(status);
        std::cerr << "[-] Failed to initialize NVML: " << (err ? err : "Unknown")
                  << " (Code: " << status << ")" << std::endl;
        return 1;
    }

    std::cout << "[+] NVML initialized successfully!" << std::endl;
    std::cout << "[+] Library loaded from: " << nvml.getLoadedPath() << std::endl;

    std::string driverVersion = nvml.getDriverVersion();
    if (!driverVersion.empty()) {
        std::cout << "[+] NVIDIA Driver Version: " << driverVersion << std::endl;
    }

    unsigned int deviceCount = nvml.getDeviceCount();
    std::cout << "[+] Total Detected GPU Count: " << deviceCount << std::endl;
    std::cout << "==================================================" << std::endl;

    // 2. Enumerate physical GPUs using GpuResource
    std::cout << "\nEnumerating available physical GPUs via GpuResource...\n" << std::endl;
    auto gpus = fgp::cuda::GpuResource::listAvailableGpus(nvml);

    if (gpus.empty()) {
        std::cout << "[!] No active NVIDIA GPUs discovered by GpuResource." << std::endl;
    } else {
        std::cout << "[+] Discovered " << gpus.size() << " physical GPU resource(s):\n" << std::endl;
        for (const auto& gpu : gpus) {
            gpu.printDetails(std::cout);
            std::cout << "--------------------------------------------------" << std::endl;
        }
    }

    // 3. Clean shutdown
    nvml.shutdown();
    std::cout << "\n[+] NVML shutdown complete. Test finished successfully." << std::endl;

    std::cout << "\nPress [Enter] to exit..." << std::endl;
    std::cin.get();
    return 0;
}
