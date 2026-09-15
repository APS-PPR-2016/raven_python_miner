#pragma once

#include "fgp/cuda/cuda_nvml.h"
#include <string>
#include <vector>
#include <cstdint>
#include <iostream>

namespace fgp::cuda {

struct GpuMemoryInfo {
    uint64_t totalBytes = 0;
    uint64_t freeBytes = 0;
    uint64_t usedBytes = 0;

    double totalMB() const { return static_cast<double>(totalBytes) / (1024.0 * 1024.0); }
    double freeMB() const { return static_cast<double>(freeBytes) / (1024.0 * 1024.0); }
    double usedMB() const { return static_cast<double>(usedBytes) / (1024.0 * 1024.0); }

    double totalGB() const { return static_cast<double>(totalBytes) / (1024.0 * 1024.0 * 1024.0); }
    double freeGB() const { return static_cast<double>(freeBytes) / (1024.0 * 1024.0 * 1024.0); }
    double usedGB() const { return static_cast<double>(usedBytes) / (1024.0 * 1024.0 * 1024.0); }
};

struct GpuUtilizationInfo {
    unsigned int gpu = 0;     // GPU compute core utilization (0-100%)
    unsigned int memory = 0;  // Memory controller bandwidth utilization (0-100%)
};

class GpuResource {
public:
    GpuResource();
    GpuResource(unsigned int index, CudaNVML* nvml);

    // Static discovery: query NVML and enumerate all available physical GPUs
    static std::vector<GpuResource> listAvailableGpus(CudaNVML& nvml);

    // Refresh dynamic telemetry: memory, utilization, temperature, power, fan speed
    bool refresh();

    // Validity check
    bool isValid() const { return m_valid; }

    // Identification & hardware info
    unsigned int getIndex() const { return m_index; }
    const std::string& getName() const { return m_name; }
    const std::string& getUuid() const { return m_uuid; }
    const std::string& getPciBusId() const { return m_pciBusId; }
    nvmlDevice_t getDeviceHandle() const { return m_deviceHandle; }

    // Memory statistics
    const GpuMemoryInfo& getMemoryInfo() const { return m_memory; }
    uint64_t getTotalMemoryBytes() const { return m_memory.totalBytes; }
    uint64_t getFreeMemoryBytes() const { return m_memory.freeBytes; }
    uint64_t getUsedMemoryBytes() const { return m_memory.usedBytes; }
    double getTotalMemoryMB() const { return m_memory.totalMB(); }
    double getFreeMemoryMB() const { return m_memory.freeMB(); }
    double getUsedMemoryMB() const { return m_memory.usedMB(); }
    double getTotalMemoryGB() const { return m_memory.totalGB(); }
    double getFreeMemoryGB() const { return m_memory.freeGB(); }
    double getUsedMemoryGB() const { return m_memory.usedGB(); }

    // Utilization & sensor telemetry
    const GpuUtilizationInfo& getUtilization() const { return m_utilization; }
    unsigned int getGpuUtilization() const { return m_utilization.gpu; }
    unsigned int getMemoryUtilization() const { return m_utilization.memory; }
    unsigned int getTemperatureC() const { return m_temperatureC; }
    unsigned int getPowerUsageMilliwatts() const { return m_powerUsageMw; }
    double getPowerUsageWatts() const { return static_cast<double>(m_powerUsageMw) / 1000.0; }
    unsigned int getPowerLimitMilliwatts() const { return m_powerLimitMw; }
    double getPowerLimitWatts() const { return static_cast<double>(m_powerLimitMw) / 1000.0; }
    unsigned int getFanSpeedPercent() const { return m_fanSpeedPercent; }

    // Formatting / print utilities
    void printDetails(std::ostream& os = std::cout) const;
    std::string toString() const;

private:
    unsigned int m_index = 0;
    CudaNVML* m_nvml = nullptr;
    nvmlDevice_t m_deviceHandle = nullptr;
    bool m_valid = false;

    // Static GPU information
    std::string m_name;
    std::string m_uuid;
    std::string m_pciBusId;

    // Dynamic GPU telemetry
    GpuMemoryInfo m_memory;
    GpuUtilizationInfo m_utilization;
    unsigned int m_temperatureC = 0;
    unsigned int m_powerUsageMw = 0;
    unsigned int m_powerLimitMw = 0;
    unsigned int m_fanSpeedPercent = 0;

    void initializeStaticInfo();
};

} // namespace fgp::cuda

using GpuResource = fgp::cuda::GpuResource;
