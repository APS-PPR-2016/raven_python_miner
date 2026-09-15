#include "fgp/cuda/gpu_resource.h"

#include <iomanip>
#include <sstream>

namespace fgp::cuda {

GpuResource::GpuResource()
    : m_index(0),
      m_nvml(nullptr),
      m_deviceHandle(nullptr),
      m_valid(false) {
}

GpuResource::GpuResource(unsigned int index, CudaNVML* nvml)
    : m_index(index),
      m_nvml(nvml),
      m_deviceHandle(nullptr),
      m_valid(false) {
    if (m_nvml && m_nvml->isInitialized()) {
        initializeStaticInfo();
    }
}

void GpuResource::initializeStaticInfo() {
    if (!m_nvml) {
        return;
    }

    nvmlReturn_t res = m_nvml->getDeviceHandleByIndex(m_index, &m_deviceHandle);
    if (res != NVML_SUCCESS || !m_deviceHandle) {
        m_valid = false;
        return;
    }

    // Query device name
    std::string name;
    if (m_nvml->getDeviceName(m_deviceHandle, name) == NVML_SUCCESS) {
        m_name = name;
    } else {
        m_name = "Unknown NVIDIA GPU";
    }

    // Query UUID
    std::string uuid;
    if (m_nvml->getDeviceUUID(m_deviceHandle, uuid) == NVML_SUCCESS) {
        m_uuid = uuid;
    }

    // Query PCI Bus ID
    nvmlPciInfo_t pci;
    if (m_nvml->getDevicePciInfo(m_deviceHandle, pci) == NVML_SUCCESS) {
        m_pciBusId = pci.busId;
    }

    // Query power limit (optional / best effort)
    unsigned int powerLimitMw = 0;
    if (m_nvml->getDevicePowerLimit(m_deviceHandle, powerLimitMw) == NVML_SUCCESS) {
        m_powerLimitMw = powerLimitMw;
    }

    m_valid = true;

    // Perform initial query of dynamic telemetry
    refresh();
}

bool GpuResource::refresh() {
    if (!m_valid || !m_nvml || !m_deviceHandle) {
        return false;
    }

    // Refresh memory info
    nvmlMemory_t mem;
    if (m_nvml->getDeviceMemoryInfo(m_deviceHandle, mem) == NVML_SUCCESS) {
        m_memory.totalBytes = mem.total;
        m_memory.freeBytes = mem.free;
        m_memory.usedBytes = mem.used;
    }

    // Refresh utilization
    nvmlUtilization_t util;
    if (m_nvml->getDeviceUtilization(m_deviceHandle, util) == NVML_SUCCESS) {
        m_utilization.gpu = util.gpu;
        m_utilization.memory = util.memory;
    }

    // Refresh temperature
    unsigned int temp = 0;
    if (m_nvml->getDeviceTemperature(m_deviceHandle, temp) == NVML_SUCCESS) {
        m_temperatureC = temp;
    }

    // Refresh power usage
    unsigned int powerMw = 0;
    if (m_nvml->getDevicePowerUsage(m_deviceHandle, powerMw) == NVML_SUCCESS) {
        m_powerUsageMw = powerMw;
    }

    // Refresh fan speed
    unsigned int fanSpeed = 0;
    if (m_nvml->getDeviceFanSpeed(m_deviceHandle, fanSpeed) == NVML_SUCCESS) {
        m_fanSpeedPercent = fanSpeed;
    }

    return true;
}

std::vector<GpuResource> GpuResource::listAvailableGpus(CudaNVML& nvml) {
    std::vector<GpuResource> devices;

    if (!nvml.isInitialized()) {
        if (nvml.initialize() != NVML_SUCCESS) {
            return devices;
        }
    }

    unsigned int count = nvml.getDeviceCount();
    devices.reserve(count);

    for (unsigned int i = 0; i < count; ++i) {
        GpuResource gpu(i, &nvml);
        if (gpu.isValid()) {
            devices.push_back(std::move(gpu));
        }
    }

    return devices;
}

std::string GpuResource::toString() const {
    std::ostringstream ss;
    printDetails(ss);
    return ss.str();
}

void GpuResource::printDetails(std::ostream& os) const {
    if (!m_valid) {
        os << "[GPU #" << m_index << "] (Invalid or unreachable device)" << std::endl;
        return;
    }

    os << "[GPU #" << m_index << "] " << m_name << std::endl;
    if (!m_pciBusId.empty()) {
        os << "    PCI Bus ID:   " << m_pciBusId << std::endl;
    }
    if (!m_uuid.empty()) {
        os << "    UUID:         " << m_uuid << std::endl;
    }

    // Memory info
    os << "    Memory:       "
       << std::fixed << std::setprecision(1) << m_memory.usedMB() << " MB / "
       << m_memory.totalMB() << " MB ("
       << m_memory.freeMB() << " MB free)" << std::endl;

    // Telemetry
    os << "    Utilization:  GPU " << m_utilization.gpu << "%, Memory " << m_utilization.memory << "%" << std::endl;
    if (m_temperatureC > 0) {
        os << "    Temperature:  " << m_temperatureC << " C" << std::endl;
    }
    if (m_powerUsageMw > 0) {
        os << "    Power Usage:  " << std::fixed << std::setprecision(1) << getPowerUsageWatts() << " W";
        if (m_powerLimitMw > 0) {
            os << " / " << getPowerLimitWatts() << " W limit";
        }
        os << std::endl;
    }
    if (m_fanSpeedPercent > 0) {
        os << "    Fan Speed:    " << m_fanSpeedPercent << "%" << std::endl;
    }
}

} // namespace fgp::cuda
