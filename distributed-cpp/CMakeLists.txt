cmake_minimum_required(VERSION 3.20)
project(raven_mining_distributed C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Platform definitions
if(MSVC)
    add_compile_options(/W3 /EHsc /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

# Global Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../raven_miner_cpp/libraries/lib-fg-cuda/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../raven_miner_cpp/libraries/lib-ethash
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# 1. Keccak Library
add_library(keccak STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../raven_miner_cpp/libraries/lib-ethash/keccak/keccak.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../raven_miner_cpp/libraries/lib-ethash/keccak/keccakf800.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../raven_miner_cpp/libraries/lib-ethash/keccak/keccakf1600.c
)
target_include_directories(keccak PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../raven_miner_cpp/libraries/lib-ethash/keccak
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# 2. FG CUDA Library (NVML dynamic discovery, temperature, power, fan & memory telemetry)
add_library(fg_cuda STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../raven_miner_cpp/libraries/lib-fg-cuda/src/cuda_nvml.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../raven_miner_cpp/libraries/lib-fg-cuda/src/gpu_resource.cpp
)
target_include_directories(fg_cuda PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../raven_miner_cpp/libraries/lib-fg-cuda/include
)

# Common engine core components
set(ENGINE_SOURCES
    src/engine_core.cpp
    src/dag_generator.cpp
)

# -------------------------------------------------------------
# Target 1: raven_mining_server
# (Follows distributed/kawpow_server.py: Stratum pool client,
#  DAG synthesizer & HTTP distribution, client coordinator,
#  share queue dispatcher)
# -------------------------------------------------------------
add_executable(raven_mining_server
    src/server_main.cpp
    ${ENGINE_SOURCES}
)
target_link_libraries(raven_mining_server PRIVATE keccak)
if(WIN32)
    target_link_libraries(raven_mining_server PRIVATE ws2_32)
endif()

# -------------------------------------------------------------
# Target 2: raven_mining_client
# (Follows distributed/kawpow_client.py: Coordinator TCP client,
#  DAG download & VRAM caching, NVML hardware monitoring,
#  multi-GPU ProgPOW search worker, real-time share submission)
# -------------------------------------------------------------
add_executable(raven_mining_client
    src/client_main.cpp
    ${ENGINE_SOURCES}
)
target_link_libraries(raven_mining_client PRIVATE fg_cuda keccak)
if(WIN32)
    target_link_libraries(raven_mining_client PRIVATE ws2_32)
endif()

# Summary
message(STATUS "Configured targets:")
message(STATUS "  - raven_mining_server (Mining coordinator & DAG distributor)")
message(STATUS "  - raven_mining_client (Multi-GPU worker & hardware monitor)")
