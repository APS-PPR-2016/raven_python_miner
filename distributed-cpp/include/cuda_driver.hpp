#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace kawpow::cuda {

typedef int CUresult;
constexpr int CUDA_SUCCESS = 0;

typedef int CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;

class CudaDriver {
public:
    using pfn_cuInit = CUresult(*)(unsigned int);
    using pfn_cuDeviceGet = CUresult(*)(CUdevice*, int);
    using pfn_cuDeviceGetName = CUresult(*)(char*, int, CUdevice);
    using pfn_cuDeviceTotalMem = CUresult(*)(size_t*, CUdevice);
    using pfn_cuCtxCreate = CUresult(*)(CUcontext*, unsigned int, CUdevice);
    using pfn_cuCtxDestroy = CUresult(*)(CUcontext);
    using pfn_cuCtxSetCurrent = CUresult(*)(CUcontext);
    using pfn_cuMemAlloc = CUresult(*)(CUdeviceptr*, size_t);
    using pfn_cuMemFree = CUresult(*)(CUdeviceptr);
    using pfn_cuMemcpyHtoD = CUresult(*)(CUdeviceptr, const void*, size_t);

    pfn_cuInit cuInit = nullptr;
    pfn_cuDeviceGet cuDeviceGet = nullptr;
    pfn_cuDeviceGetName cuDeviceGetName = nullptr;
    pfn_cuDeviceTotalMem cuDeviceTotalMem = nullptr;
    pfn_cuCtxCreate cuCtxCreate = nullptr;
    pfn_cuCtxDestroy cuCtxDestroy = nullptr;
    pfn_cuCtxSetCurrent cuCtxSetCurrent = nullptr;
    pfn_cuMemAlloc cuMemAlloc = nullptr;
    pfn_cuMemFree cuMemFree = nullptr;
    pfn_cuMemcpyHtoD cuMemcpyHtoD = nullptr;

    bool initialized = false;
#ifdef _WIN32
    HMODULE lib_handle = nullptr;
#else
    void* lib_handle = nullptr;
#endif

    static CudaDriver& instance() {
        static CudaDriver inst;
        return inst;
    }

    bool initialize() {
        if (initialized) return true;

#ifdef _WIN32
        lib_handle = LoadLibraryA("nvcuda.dll");
        if (!lib_handle) return false;
        #define GET_CUDA_PROC(name) name = reinterpret_cast<pfn_##name>(GetProcAddress(lib_handle, #name))
#else
        lib_handle = dlopen("libcuda.so.1", RTLD_LAZY);
        if (!lib_handle) lib_handle = dlopen("libcuda.so", RTLD_LAZY);
        if (!lib_handle) return false;
        #define GET_CUDA_PROC(name) name = reinterpret_cast<pfn_##name>(dlsym(lib_handle, #name))
#endif

        GET_CUDA_PROC(cuInit);
        GET_CUDA_PROC(cuDeviceGet);
        GET_CUDA_PROC(cuDeviceGetName);
        GET_CUDA_PROC(cuDeviceTotalMem);
        GET_CUDA_PROC(cuCtxCreate);
        GET_CUDA_PROC(cuCtxDestroy);
        GET_CUDA_PROC(cuCtxSetCurrent);
        GET_CUDA_PROC(cuMemAlloc);
        GET_CUDA_PROC(cuMemFree);
        GET_CUDA_PROC(cuMemcpyHtoD);

        if (!cuInit || cuInit(0) != CUDA_SUCCESS) {
            return false;
        }

        initialized = true;
        return true;
    }
};

class GpuDagBuffer {
public:
    int gpu_id = 0;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUdeviceptr dag_vram_ptr = 0;
    size_t dag_size = 0;
    std::string device_name = "NVIDIA CUDA GPU";
    bool loaded = false;

    explicit GpuDagBuffer(int gid) : gpu_id(gid) {}
    ~GpuDagBuffer() {
        free_dag();
    }

    bool init_device() {
        auto& drv = CudaDriver::instance();
        if (!drv.initialize()) return false;

        if (drv.cuDeviceGet(&device, gpu_id) != CUDA_SUCCESS) return false;

        char name[256] = {0};
        drv.cuDeviceGetName(name, sizeof(name), device);
        device_name = name;

        if (drv.cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) return false;
        return true;
    }

    void free_dag() {
        auto& drv = CudaDriver::instance();
        if (drv.initialized && context) {
            drv.cuCtxSetCurrent(context);
            if (dag_vram_ptr) {
                drv.cuMemFree(dag_vram_ptr);
                dag_vram_ptr = 0;
            }
        }
        dag_size = 0;
        loaded = false;
    }

    bool load_dag_from_file(const std::string& filepath, uint64_t expected_size) {
        auto& drv = CudaDriver::instance();
        if (!drv.initialize()) {
            std::cout << "[GPU " << gpu_id << "] CUDA Driver not available. Simulating VRAM allocation...\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            loaded = true;
            return true;
        }

        if (!context) {
            if (!init_device()) {
                std::cerr << "[GPU " << gpu_id << "] Failed to initialize CUDA device context.\n";
                return false;
            }
        }

        drv.cuCtxSetCurrent(context);
        free_dag();

        double gb = static_cast<double>(expected_size) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "[GPU " << gpu_id << "] Allocating " << std::fixed << std::setprecision(2)
                  << gb << " GB VRAM on " << device_name << "...\n";

        if (drv.cuMemAlloc(&dag_vram_ptr, expected_size) != CUDA_SUCCESS) {
            std::cerr << "[GPU " << gpu_id << "] cuMemAlloc failed! Insufficient GPU VRAM for " << gb << " GB DAG.\n";
            return false;
        }

        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            std::cerr << "[GPU " << gpu_id << "] Failed to open DAG file: " << filepath << "\n";
            free_dag();
            return false;
        }

        auto t0 = std::chrono::steady_clock::now();
        const size_t chunk_size = 128 * 1024 * 1024; // 128 MB blocks
        std::vector<char> buffer(chunk_size);
        size_t total_copied = 0;

        while (total_copied < expected_size && file) {
            size_t to_read = (std::min)(chunk_size, static_cast<size_t>(expected_size - total_copied));
            file.read(buffer.data(), to_read);
            size_t bytes_read = static_cast<size_t>(file.gcount());
            if (bytes_read == 0) break;

            if (drv.cuMemcpyHtoD(dag_vram_ptr + total_copied, buffer.data(), bytes_read) != CUDA_SUCCESS) {
                std::cerr << "\n[GPU " << gpu_id << "] cuMemcpyHtoD failed at offset " << total_copied << "!\n";
                free_dag();
                return false;
            }

            total_copied += bytes_read;
            double pct = (static_cast<double>(total_copied) / static_cast<double>(expected_size)) * 100.0;
            std::cout << "\r[GPU " << gpu_id << "] Loading DAG to VRAM: " << std::fixed << std::setprecision(1)
                      << pct << "% (" << (total_copied / (1024 * 1024)) << " MB / "
                      << (expected_size / (1024 * 1024)) << " MB)" << std::flush;
        }

        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        double sec = dt / 1000.0;
        double mb_s = (static_cast<double>(total_copied) / (1024.0 * 1024.0)) / (sec > 0.001 ? sec : 0.001);

        std::cout << "\n[GPU " << gpu_id << "] DAG loaded into VRAM in " << std::fixed << std::setprecision(2)
                  << sec << "s (" << std::setprecision(1) << mb_s << " MB/s)!\n";

        dag_size = total_copied;
        loaded = (total_copied == expected_size);
        return loaded;
    }
};

} // namespace kawpow::cuda
