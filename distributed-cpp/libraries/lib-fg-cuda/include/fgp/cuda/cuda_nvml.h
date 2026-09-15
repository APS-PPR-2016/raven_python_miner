#pragma once

#include <string>
#include <vector>

// NVML return codes
enum nvmlReturn_t {
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_INSUFFICIENT_POWER = 8,
    NVML_ERROR_DRIVER_NOT_LOADED = 9,
    NVML_ERROR_TIMEOUT = 10,
    NVML_ERROR_IRQ_ISSUE = 11,
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,
    NVML_ERROR_CORRUPTED_INFOROM = 14,
    NVML_ERROR_GPU_IS_LOST = 15,
    NVML_ERROR_RESET_REQUIRED = 16,
    NVML_ERROR_OPERATING_SYSTEM = 17,
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,
    NVML_ERROR_IN_USE = 19,
    NVML_ERROR_UNKNOWN = 999
};

// NVML device handle type
typedef struct nvmlDevice_st* nvmlDevice_t;

// NVML memory info struct
typedef struct nvmlMemory_st {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

// NVML utilization struct
typedef struct nvmlUtilization_st {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

// NVML temperature sensor enum
enum nvmlTemperatureSensors_t {
    NVML_TEMPERATURE_GPU = 0
};

// NVML PCI info struct
typedef struct nvmlPciInfo_st {
    char busIdLegacy[16];
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int pciDeviceId;
    unsigned int pciSubSystemId;
    char busId[32];
} nvmlPciInfo_t;

// Function pointer signatures for NVML APIs
typedef nvmlReturn_t (*nvmlInit_t)();
typedef nvmlReturn_t (*nvmlShutdown_t)();
typedef nvmlReturn_t (*nvmlDeviceGetCount_t)(unsigned int*);
typedef nvmlReturn_t (*nvmlSystemGetDriverVersion_t)(char*, unsigned int);
typedef const char*  (*nvmlErrorString_t)(nvmlReturn_t);

// Device-level function pointer signatures
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_t)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetName_t)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*nvmlDeviceGetUUID_t)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*nvmlDeviceGetPciInfo_t)(nvmlDevice_t, nvmlPciInfo_t*);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_t)(nvmlDevice_t, nvmlMemory_t*);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_t)(nvmlDevice_t, nvmlUtilization_t*);
typedef nvmlReturn_t (*nvmlDeviceGetTemperature_t)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetPowerUsage_t)(nvmlDevice_t, unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetEnforcedPowerLimit_t)(nvmlDevice_t, unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetFanSpeed_t)(nvmlDevice_t, unsigned int*);

namespace fgp::cuda {

class CudaNVML {
public:
    CudaNVML();
    explicit CudaNVML(bool autoInit);
    ~CudaNVML();

    // Non-copyable (manages dynamic library handle)
    CudaNVML(const CudaNVML&) = delete;
    CudaNVML& operator=(const CudaNVML&) = delete;

    // Movable
    CudaNVML(CudaNVML&& other) noexcept;
    CudaNVML& operator=(CudaNVML&& other) noexcept;

    // Dynamic library management
    bool load(const std::string& customPath = "");
    void unload();
    bool isLoaded() const;

    // NVML initialization & shutdown
    nvmlReturn_t initialize();
    nvmlReturn_t shutdown();
    bool isInitialized() const;

    // System-level queries
    nvmlReturn_t getDeviceCount(unsigned int& count) const;
    unsigned int getDeviceCount() const;

    nvmlReturn_t getDriverVersion(std::string& version) const;
    std::string getDriverVersion() const;

    const char* getErrorString(nvmlReturn_t result) const;
    const std::string& getLoadedPath() const;

    // Device-level queries
    nvmlReturn_t getDeviceHandleByIndex(unsigned int index, nvmlDevice_t* device) const;
    nvmlReturn_t getDeviceName(nvmlDevice_t device, std::string& name) const;
    nvmlReturn_t getDeviceUUID(nvmlDevice_t device, std::string& uuid) const;
    nvmlReturn_t getDevicePciInfo(nvmlDevice_t device, nvmlPciInfo_t& pci) const;
    nvmlReturn_t getDeviceMemoryInfo(nvmlDevice_t device, nvmlMemory_t& memory) const;
    nvmlReturn_t getDeviceUtilization(nvmlDevice_t device, nvmlUtilization_t& utilization) const;
    nvmlReturn_t getDeviceTemperature(nvmlDevice_t device, unsigned int& temp) const;
    nvmlReturn_t getDevicePowerUsage(nvmlDevice_t device, unsigned int& powerMilliwatts) const;
    nvmlReturn_t getDevicePowerLimit(nvmlDevice_t device, unsigned int& limitMilliwatts) const;
    nvmlReturn_t getDeviceFanSpeed(nvmlDevice_t device, unsigned int& fanSpeed) const;

    // Diagnostic check (replaces original main check behavior)
    int check();
    static int runCheck();

private:
    void* m_nvmlLib = nullptr;
    std::string m_loadedPath;
    bool m_initialized = false;

    // Resolved system function pointers
    nvmlInit_t m_nvmlInit = nullptr;
    nvmlShutdown_t m_nvmlShutdown = nullptr;
    nvmlDeviceGetCount_t m_nvmlDeviceGetCount = nullptr;
    nvmlSystemGetDriverVersion_t m_nvmlSystemGetDriverVersion = nullptr;
    nvmlErrorString_t m_nvmlErrorString = nullptr;

    // Resolved device function pointers
    nvmlDeviceGetHandleByIndex_t m_nvmlDeviceGetHandleByIndex = nullptr;
    nvmlDeviceGetName_t m_nvmlDeviceGetName = nullptr;
    nvmlDeviceGetUUID_t m_nvmlDeviceGetUUID = nullptr;
    nvmlDeviceGetPciInfo_t m_nvmlDeviceGetPciInfo = nullptr;
    nvmlDeviceGetMemoryInfo_t m_nvmlDeviceGetMemoryInfo = nullptr;
    nvmlDeviceGetUtilizationRates_t m_nvmlDeviceGetUtilizationRates = nullptr;
    nvmlDeviceGetTemperature_t m_nvmlDeviceGetTemperature = nullptr;
    nvmlDeviceGetPowerUsage_t m_nvmlDeviceGetPowerUsage = nullptr;
    nvmlDeviceGetEnforcedPowerLimit_t m_nvmlDeviceGetEnforcedPowerLimit = nullptr;
    nvmlDeviceGetFanSpeed_t m_nvmlDeviceGetFanSpeed = nullptr;

    std::vector<std::string> getDefaultLibraryCandidates() const;
    bool resolveSymbols();
    void resetSymbols();
};

} // namespace fgp::cuda

using CudaNVML = fgp::cuda::CudaNVML;
