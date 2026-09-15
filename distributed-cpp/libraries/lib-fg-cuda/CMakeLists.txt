cmake_minimum_required(VERSION 3.10)
project(fg-cuda CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(fg-cuda STATIC
    src/cuda_nvml.cpp
    src/gpu_resource.cpp
)
add_library(fg::cuda ALIAS fg-cuda)

target_include_directories(fg-cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(WIN32)
    target_link_libraries(fg-cuda PUBLIC fg-big-hash fg-big-uint-win64)
elseif(UNIX)
    target_link_libraries(fg-cuda PUBLIC fg-big-hash fg-big-uint-elf64)
endif()

# Test project
add_executable(test-gpu-resource
    tests/test_gpu_resource.cpp
)
target_link_libraries(test-gpu-resource PRIVATE fg-cuda)