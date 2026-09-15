cmake_minimum_required(VERSION 3.10)

project(ethash-cl CXX C)

set(ETHASH_CL_SOURCES
    ethash-cl.cpp
)

find_package(OpenCL QUIET)

add_library(ethash-cl ${ETHASH_CL_SOURCES})

target_include_directories(ethash-cl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ethash-cl PRIVATE ethash keccak)

if(OpenCL_FOUND)
    message(STATUS "OpenCL found: ${OpenCL_LIBRARIES}")
    target_compile_definitions(ethash-cl PRIVATE HAVE_OPENCL=1)
    target_include_directories(ethash-cl PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(ethash-cl PRIVATE ${OpenCL_LIBRARIES})
else()
    message(STATUS "OpenCL not found; ethash-cl built with CPU reference fallback engine.")
endif()

if(MSVC)
    target_compile_options(ethash-cl PRIVATE /FS)
endif()
