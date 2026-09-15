#include "fgp/cuda/cuda_nvml.h"

#include <iostream>
#include <vector>
#include <cstring>
#include <utility>

#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #define LOAD_LIBRARY(path) LoadLibraryA(path)
    #define GET_SYMBOL(mod, name) GetProcAddress(static_cast<HMODULE>(mod), name)
    #define FREE_LIBRARY(mod) FreeLibrary(static_cast<HMODULE>(mod))
#else
    #include <dlfcn.h>
    #define LOAD_LIBRARY(path) dlopen(path, RTLD_NOW)
    #define GET_SYMBOL(mod, name) dlsym(mod, name)
    #define FREE_LIBRARY(mod) dlclose(mod)
#endif

namespace fgp::cuda {

CudaNVML::CudaNVML() {
}

CudaNVML::CudaNVML(bool autoInit) {
    if (autoInit) {
        initialize();
    }
}

CudaNVML::~CudaNVML() {
    unload();
}

CudaNVML::CudaNVML(CudaNVML&& other) noexcept
    : m_nvmlLib(other.m_nvmlLib),
      m_loadedPath(std::move(other.m_loadedPath)),
      m_initialized(other.m_initialized),
      m_nvmlInit(other.m_nvmlInit),
      m_nvmlShutdown(other.m_nvmlShutdown),
      m_nvmlDeviceGetCount(other.m_nvmlDeviceGetCount),
      m_nvmlSystemGetDriverVersion(other.m_nvmlSystemGetDriverVersion),
      m_nvmlErrorString(other.m_nvmlErrorString),
      m_nvmlDeviceGetHandleByIndex(other.m_nvmlDeviceGetHandleByIndex),
      m_nvmlDeviceGetName(other.m_nvmlDeviceGetName),
      m_nvmlDeviceGetUUID(other.m_nvmlDeviceGetUUID),
      m_nvmlDeviceGetPciInfo(other.m_nvmlDeviceGetPciInfo),
      m_nvmlDeviceGetMemoryInfo(other.m_nvmlDeviceGetMemoryInfo),
      m_nvmlDeviceGetUtilizationRates(other.m_nvmlDeviceGetUtilizationRates),
      m_nvmlDeviceGetTemperature(other.m_nvmlDeviceGetTemperature),
      m_nvmlDeviceGetPowerUsage(other.m_nvmlDeviceGetPowerUsage),
      m_nvmlDeviceGetEnforcedPowerLimit(other.m_nvmlDeviceGetEnforcedPowerLimit),
      m_nvmlDeviceGetFanSpeed(other.m_nvmlDeviceGetFanSpeed) {
    other.m_nvmlLib = nullptr;
    other.m_initialized = false;
    other.resetSymbols();
}

CudaNVML& CudaNVML::operator=(CudaNVML&& other) noexcept {
    if (this != &other) {
        unload();
        m_nvmlLib = other.m_nvmlLib;
        m_loadedPath = std::move(other.m_loadedPath);
        m_initialized = other.m_initialized;
        m_nvmlInit = other.m_nvmlInit;
        m_nvmlShutdown = other.m_nvmlShutdown;
        m_nvmlDeviceGetCount = other.m_nvmlDeviceGetCount;
        m_nvmlSystemGetDriverVersion = other.m_nvmlSystemGetDriverVersion;
        m_nvmlErrorString = other.m_nvmlErrorString;

        m_nvmlDeviceGetHandleByIndex = other.m_nvmlDeviceGetHandleByIndex;
        m_nvmlDeviceGetName = other.m_nvmlDeviceGetName;
        m_nvmlDeviceGetUUID = other.m_nvmlDeviceGetUUID;
        m_nvmlDeviceGetPciInfo = other.m_nvmlDeviceGetPciInfo;
        m_nvmlDeviceGetMemoryInfo = other.m_nvmlDeviceGetMemoryInfo;
        m_nvmlDeviceGetUtilizationRates = other.m_nvmlDeviceGetUtilizationRates;
        m_nvmlDeviceGetTemperature = other.m_nvmlDeviceGetTemperature;
        m_nvmlDeviceGetPowerUsage = other.m_nvmlDeviceGetPowerUsage;
        m_nvmlDeviceGetEnforcedPowerLimit = other.m_nvmlDeviceGetEnforcedPowerLimit;
        m_nvmlDeviceGetFanSpeed = other.m_nvmlDeviceGetFanSpeed;

        other.m_nvmlLib = nullptr;
        other.m_initialized = false;
        other.resetSymbols();
    }
    return *this;
}

std::vector<std::string> CudaNVML::getDefaultLibraryCandidates() const {
#if defined(_WIN32) || defined(_WIN64)
    return {
        "nvml.dll",
        "C:\\Windows\\System32\\nvml.dll",
        "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll"
    };
#else
    return {
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib64/libnvidia-ml.so.1"
    };
#endif
}

void CudaNVML::resetSymbols() {
    m_nvmlInit = nullptr;
    m_nvmlShutdown = nullptr;
    m_nvmlDeviceGetCount = nullptr;
    m_nvmlSystemGetDriverVersion = nullptr;
    m_nvmlErrorString = nullptr;

    m_nvmlDeviceGetHandleByIndex = nullptr;
    m_nvmlDeviceGetName = nullptr;
    m_nvmlDeviceGetUUID = nullptr;
    m_nvmlDeviceGetPciInfo = nullptr;
    m_nvmlDeviceGetMemoryInfo = nullptr;
    m_nvmlDeviceGetUtilizationRates = nullptr;
    m_nvmlDeviceGetTemperature = nullptr;
    m_nvmlDeviceGetPowerUsage = nullptr;
    m_nvmlDeviceGetEnforcedPowerLimit = nullptr;
    m_nvmlDeviceGetFanSpeed = nullptr;
}

bool CudaNVML::resolveSymbols() {
    if (!m_nvmlLib) {
        return false;
    }

    // Resolve system function symbols
    m_nvmlInit = reinterpret_cast<nvmlInit_t>(GET_SYMBOL(m_nvmlLib, "nvmlInit_v2"));
    if (!m_nvmlInit) {
        m_nvmlInit = reinterpret_cast<nvmlInit_t>(GET_SYMBOL(m_nvmlLib, "nvmlInit"));
    }

    m_nvmlShutdown = reinterpret_cast<nvmlShutdown_t>(GET_SYMBOL(m_nvmlLib, "nvmlShutdown"));
    m_nvmlDeviceGetCount = reinterpret_cast<nvmlDeviceGetCount_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetCount_v2"));
    if (!m_nvmlDeviceGetCount) {
        m_nvmlDeviceGetCount = reinterpret_cast<nvmlDeviceGetCount_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetCount"));
    }

    m_nvmlSystemGetDriverVersion = reinterpret_cast<nvmlSystemGetDriverVersion_t>(GET_SYMBOL(m_nvmlLib, "nvmlSystemGetDriverVersion"));
    m_nvmlErrorString = reinterpret_cast<nvmlErrorString_t>(GET_SYMBOL(m_nvmlLib, "nvmlErrorString"));

    // Required core system symbols
    if (!m_nvmlInit || !m_nvmlShutdown || !m_nvmlDeviceGetCount) {
        resetSymbols();
        return false;
    }

    // Resolve device function symbols
    m_nvmlDeviceGetHandleByIndex = reinterpret_cast<nvmlDeviceGetHandleByIndex_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetHandleByIndex_v2"));
    if (!m_nvmlDeviceGetHandleByIndex) {
        m_nvmlDeviceGetHandleByIndex = reinterpret_cast<nvmlDeviceGetHandleByIndex_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetHandleByIndex"));
    }

    m_nvmlDeviceGetName = reinterpret_cast<nvmlDeviceGetName_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetName"));
    m_nvmlDeviceGetUUID = reinterpret_cast<nvmlDeviceGetUUID_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetUUID"));

    m_nvmlDeviceGetPciInfo = reinterpret_cast<nvmlDeviceGetPciInfo_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetPciInfo_v3"));
    if (!m_nvmlDeviceGetPciInfo) {
        m_nvmlDeviceGetPciInfo = reinterpret_cast<nvmlDeviceGetPciInfo_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetPciInfo"));
    }

    m_nvmlDeviceGetMemoryInfo = reinterpret_cast<nvmlDeviceGetMemoryInfo_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetMemoryInfo"));
    m_nvmlDeviceGetUtilizationRates = reinterpret_cast<nvmlDeviceGetUtilizationRates_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetUtilizationRates"));
    m_nvmlDeviceGetTemperature = reinterpret_cast<nvmlDeviceGetTemperature_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetTemperature"));
    m_nvmlDeviceGetPowerUsage = reinterpret_cast<nvmlDeviceGetPowerUsage_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetPowerUsage"));
    
    m_nvmlDeviceGetEnforcedPowerLimit = reinterpret_cast<nvmlDeviceGetEnforcedPowerLimit_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetEnforcedPowerLimit"));
    if (!m_nvmlDeviceGetEnforcedPowerLimit) {
        m_nvmlDeviceGetEnforcedPowerLimit = reinterpret_cast<nvmlDeviceGetEnforcedPowerLimit_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetPowerManagementLimit"));
    }

    m_nvmlDeviceGetFanSpeed = reinterpret_cast<nvmlDeviceGetFanSpeed_t>(GET_SYMBOL(m_nvmlLib, "nvmlDeviceGetFanSpeed"));

    return true;
}

bool CudaNVML::load(const std::string& customPath) {
    if (m_nvmlLib) {
        return true;
    }

    std::vector<std::string> candidates;
    if (!customPath.empty()) {
        candidates.push_back(customPath);
    } else {
        candidates = getDefaultLibraryCandidates();
    }

    for (const auto& path : candidates) {
        void* lib = static_cast<void*>(LOAD_LIBRARY(path.c_str()));
        if (lib) {
            m_nvmlLib = lib;
            m_loadedPath = path;
            break;
        }
    }

    if (!m_nvmlLib) {
        return false;
    }

    if (!resolveSymbols()) {
        FREE_LIBRARY(m_nvmlLib);
        m_nvmlLib = nullptr;
        m_loadedPath.clear();
        return false;
    }

    return true;
}

void CudaNVML::unload() {
    if (m_initialized) {
        shutdown();
    }
    if (m_nvmlLib) {
        FREE_LIBRARY(m_nvmlLib);
        m_nvmlLib = nullptr;
    }
    m_loadedPath.clear();
    resetSymbols();
}

bool CudaNVML::isLoaded() const {
    return m_nvmlLib != nullptr;
}

nvmlReturn_t CudaNVML::initialize() {
    if (m_initialized) {
        return NVML_SUCCESS;
    }

    if (!isLoaded()) {
        if (!load()) {
            return NVML_ERROR_LIBRARY_NOT_FOUND;
        }
    }

    if (!m_nvmlInit) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }

    nvmlReturn_t res = m_nvmlInit();
    if (res == NVML_SUCCESS || res == NVML_ERROR_ALREADY_INITIALIZED) {
        m_initialized = true;
        return NVML_SUCCESS;
    }

    return res;
}

nvmlReturn_t CudaNVML::shutdown() {
    if (!m_initialized) {
        return NVML_SUCCESS;
    }

    nvmlReturn_t res = NVML_SUCCESS;
    if (m_nvmlShutdown) {
        res = m_nvmlShutdown();
    }
    m_initialized = false;
    return res;
}

bool CudaNVML::isInitialized() const {
    return m_initialized;
}

nvmlReturn_t CudaNVML::getDeviceCount(unsigned int& count) const {
    count = 0;
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!m_nvmlDeviceGetCount) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return m_nvmlDeviceGetCount(&count);
}

unsigned int CudaNVML::getDeviceCount() const {
    unsigned int count = 0;
    if (getDeviceCount(count) == NVML_SUCCESS) {
        return count;
    }
    return 0;
}

nvmlReturn_t CudaNVML::getDriverVersion(std::string& version) const {
    version.clear();
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!m_nvmlSystemGetDriverVersion) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    char driverVersion[80] = {0};
    nvmlReturn_t res = m_nvmlSystemGetDriverVersion(driverVersion, sizeof(driverVersion));
    if (res == NVML_SUCCESS) {
        version = driverVersion;
    }
    return res;
}

std::string CudaNVML::getDriverVersion() const {
    std::string ver;
    getDriverVersion(ver);
    return ver;
}

const char* CudaNVML::getErrorString(nvmlReturn_t result) const {
    if (m_nvmlErrorString) {
        return m_nvmlErrorString(result);
    }
    switch (result) {
        case NVML_SUCCESS: return "Success";
        case NVML_ERROR_UNINITIALIZED: return "Uninitialized";
        case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
        case NVML_ERROR_NOT_SUPPORTED: return "Not Supported";
        case NVML_ERROR_NO_PERMISSION: return "No Permission";
        case NVML_ERROR_ALREADY_INITIALIZED: return "Already Initialized";
        case NVML_ERROR_NOT_FOUND: return "Not Found";
        case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient Size";
        case NVML_ERROR_INSUFFICIENT_POWER: return "Insufficient Power";
        case NVML_ERROR_DRIVER_NOT_LOADED: return "Driver Not Loaded";
        case NVML_ERROR_TIMEOUT: return "Timeout";
        case NVML_ERROR_IRQ_ISSUE: return "IRQ Issue";
        case NVML_ERROR_LIBRARY_NOT_FOUND: return "Library Not Found";
        case NVML_ERROR_FUNCTION_NOT_FOUND: return "Function Not Found";
        case NVML_ERROR_CORRUPTED_INFOROM: return "Corrupted InfoROM";
        case NVML_ERROR_GPU_IS_LOST: return "GPU Is Lost";
        case NVML_ERROR_RESET_REQUIRED: return "Reset Required";
        case NVML_ERROR_OPERATING_SYSTEM: return "Operating System Error";
        case NVML_ERROR_LIB_RM_VERSION_MISMATCH: return "RM Version Mismatch";
        case NVML_ERROR_IN_USE: return "In Use";
        case NVML_ERROR_UNKNOWN: return "Unknown Error";
        default: return "Unknown Error";
    }
}

const std::string& CudaNVML::getLoadedPath() const {
    return m_loadedPath;
}

nvmlReturn_t CudaNVML::getDeviceHandleByIndex(unsigned int index, nvmlDevice_t* device) const {
    if (!device) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *device = nullptr;
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!m_nvmlDeviceGetHandleByIndex) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return m_nvmlDeviceGetHandleByIndex(index, device);
}

nvmlReturn_t CudaNVML::getDeviceName(nvmlDevice_t device, std::string& name) const {
    name.clear();
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!device) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!m_nvmlDeviceGetName) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    char buf[256] = {0};
    nvmlReturn_t res = m_nvmlDeviceGetName(device, buf, sizeof(buf));
    if (res == NVML_SUCCESS) {
        name = buf;
    }
    return res;
}

nvmlReturn_t CudaNVML::getDeviceUUID(nvmlDevice_t device, std::string& uuid) const {
    uuid.clear();
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!device) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!m_nvmlDeviceGetUUID) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    char buf[256] = {0};
    nvmlReturn_t res = m_nvmlDeviceGetUUID(device, buf, sizeof(buf));
    if (res == NVML_SUCCESS) {
        uuid = buf;
    }
    return res;
}

nvmlReturn_t CudaNVML::getDevicePciInfo(nvmlDevice_t device, nvmlPciInfo_t& pci) const {
    std::memset(&pci, 0, sizeof(pci));
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!device) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!m_nvmlDeviceGetPciInfo) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return m_nvmlDeviceGetPciInfo(device, &pci);
}

nvmlReturn_t CudaNVML::getDeviceMemoryInfo(nvmlDevice_t device, nvmlMemory_t& memory) const {
    std::memset(&memory, 0, sizeof(memory));
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!device) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!m_nvmlDeviceGetMemoryInfo) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return m_nvmlDeviceGetMemoryInfo(device, &memory);
}

nvmlReturn_t CudaNVML::getDeviceUtilization(nvmlDevice_t device, nvmlUtilization_t& utilization) const {
    std::memset(&utilization, 0, sizeof(utilization));
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!device) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!m_nvmlDeviceGetUtilizationRates) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return m_nvmlDeviceGetUtilizationRates(device, &utilization);
}

nvmlReturn_t CudaNVML::getDeviceTemperature(nvmlDevice_t device, unsigned int& temp) const {
    temp = 0;
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!device) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!m_nvmlDeviceGetTemperature) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return m_nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
}

nvmlReturn_t CudaNVML::getDevicePowerUsage(nvmlDevice_t device, unsigned int& powerMilliwatts) const {
    powerMilliwatts = 0;
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!device) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!m_nvmlDeviceGetPowerUsage) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return m_nvmlDeviceGetPowerUsage(device, &powerMilliwatts);
}

nvmlReturn_t CudaNVML::getDevicePowerLimit(nvmlDevice_t device, unsigned int& limitMilliwatts) const {
    limitMilliwatts = 0;
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!device) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!m_nvmlDeviceGetEnforcedPowerLimit) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return m_nvmlDeviceGetEnforcedPowerLimit(device, &limitMilliwatts);
}

nvmlReturn_t CudaNVML::getDeviceFanSpeed(nvmlDevice_t device, unsigned int& fanSpeed) const {
    fanSpeed = 0;
    if (!m_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (!device) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!m_nvmlDeviceGetFanSpeed) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    return m_nvmlDeviceGetFanSpeed(device, &fanSpeed);
}

int CudaNVML::check() {
    if (!load()) {
        std::cerr << "[-] NVML is NOT installed or could not be loaded." << std::endl;
        std::cerr << "    Reason: Driver library file not found or symbols failed to resolve." << std::endl;
        return 1;
    }

    std::cout << "[+] NVML library detected: " << m_loadedPath << std::endl;

    nvmlReturn_t res = initialize();
    if (res != NVML_SUCCESS) {
        const char* err = getErrorString(res);
        std::cerr << "[-] Failed to initialize NVML: " << (err ? err : "Unknown error")
                  << " (Code: " << res << ")" << std::endl;
        unload();
        return 2;
    }

    std::cout << "[+] NVML initialized successfully!" << std::endl;

    std::string driverVersion = getDriverVersion();
    if (!driverVersion.empty()) {
        std::cout << "    Driver Version: " << driverVersion << std::endl;
    }

    unsigned int deviceCount = 0;
    if (getDeviceCount(deviceCount) == NVML_SUCCESS) {
        std::cout << "    NVIDIA GPU Count: " << deviceCount << std::endl;
    }

    shutdown();
    unload();
    return 0;
}

int CudaNVML::runCheck() {
    CudaNVML checker;
    return checker.check();
}

} // namespace fgp::cuda
