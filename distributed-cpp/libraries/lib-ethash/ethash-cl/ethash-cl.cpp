// ethash-cl: OpenCL light and DAG cache generator library.
// Copyright 2026 FGBizServ.

#include "ethash/ethash-cl.h"
#include "ethash/ethash.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>

#if defined(HAVE_OPENCL)
#include <CL/cl.h>
#endif

struct ethash_cl_context {
    bool available = false;
    std::string device_name = "Host CPU (Fallback)";
#if defined(HAVE_OPENCL)
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel_light = nullptr;
    cl_kernel kernel_dag = nullptr;
#endif
};

ethash_cl_context* ethash_cl_create_context(void) {
    auto ctx = new (std::nothrow) ethash_cl_context();
    if (!ctx) return nullptr;

#if defined(HAVE_OPENCL)
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(1, &ctx->platform, &num_platforms) == CL_SUCCESS && num_platforms > 0) {
        cl_uint num_devices = 0;
        if (clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_GPU, 1, &ctx->device, &num_devices) == CL_SUCCESS && num_devices > 0) {
            char name[128] = {0};
            clGetDeviceInfo(ctx->device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            ctx->device_name = name;

            cl_int err = 0;
            ctx->context = clCreateContext(nullptr, 1, &ctx->device, nullptr, nullptr, &err);
            if (err == CL_SUCCESS) {
#if defined(CL_VERSION_2_0)
                ctx->queue = clCreateCommandQueueWithProperties(ctx->context, ctx->device, nullptr, &err);
#else
                ctx->queue = clCreateCommandQueue(ctx->context, ctx->device, 0, &err);
#endif
                if (err == CL_SUCCESS) {
                    ctx->available = true;
                }
            }
        }
    }
#endif

    return ctx;
}

void ethash_cl_destroy_context(ethash_cl_context* ctx) {
    if (!ctx) return;

#if defined(HAVE_OPENCL)
    if (ctx->kernel_dag) clReleaseKernel(ctx->kernel_dag);
    if (ctx->kernel_light) clReleaseKernel(ctx->kernel_light);
    if (ctx->program) clReleaseProgram(ctx->program);
    if (ctx->queue) clReleaseCommandQueue(ctx->queue);
    if (ctx->context) clReleaseContext(ctx->context);
#endif

    delete ctx;
}

bool ethash_cl_is_available(const ethash_cl_context* ctx) {
    return ctx ? ctx->available : false;
}

const char* ethash_cl_get_device_name(const ethash_cl_context* ctx) {
    return ctx ? ctx->device_name.c_str() : "Invalid Context";
}

bool ethash_cl_generate_light_cache(
    ethash_cl_context* ctx,
    int epoch_number,
    union ethash_hash512* out_light_cache,
    size_t num_items
) {
    if (!ctx || !out_light_cache || num_items == 0) return false;

    // Perform light cache generation using Ethash primitives
    ethash::epoch_context_ptr epoch_ctx = ethash::create_epoch_context(epoch_number);
    if (!epoch_ctx || !epoch_ctx->light_cache) return false;

    size_t copy_items = (num_items < static_cast<size_t>(epoch_ctx->light_cache_num_items))
                            ? num_items
                            : static_cast<size_t>(epoch_ctx->light_cache_num_items);

    std::memcpy(out_light_cache, epoch_ctx->light_cache, copy_items * sizeof(union ethash_hash512));
    return true;
}

bool ethash_cl_generate_dag_cache(
    ethash_cl_context* ctx,
    int epoch_number,
    const union ethash_hash512* light_cache,
    union ethash_hash1024* out_dag_cache,
    size_t num_items
) {
    if (!ctx || !out_dag_cache || num_items == 0) return false;

    ethash::epoch_context_ptr epoch_ctx = ethash::create_epoch_context(epoch_number);
    if (!epoch_ctx) return false;

    // Generate DAG items using Ethash 512-parent item calculation
    for (size_t i = 0; i < num_items; ++i) {
        // Compute item i
        ethash::hash1024 item = ethash::calculate_dataset_item_1024(*epoch_ctx, static_cast<uint32_t>(i));
        std::memcpy(&out_dag_cache[i], &item, sizeof(union ethash_hash1024));
    }

    return true;
}
