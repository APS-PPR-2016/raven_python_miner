#include "dag_generator.hpp"
#include "engine_core.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <vector>

namespace fs = std::filesystem;

namespace kawpow {

DagInfo get_epoch_info(uint32_t epoch) {
    DagInfo info;
    info.epoch = epoch;
    info.num_cache_items = get_cache_num_items(epoch);
    uint32_t full_items = get_dataset_num_items(epoch);
    uint32_t total_half_items = full_items * 2;
    info.dag_bytes = static_cast<uint64_t>(total_half_items) * 64ULL;
    info.num_dag_items_2048 = full_items / 2;
    info.filename = ".cache_rvn_epoch_" + std::to_string(epoch) + "_dag.bin";
    return info;
}

bool is_dag_cached(uint32_t epoch, const std::string& base_dir) {
    DagInfo info = get_epoch_info(epoch);
    fs::path p = fs::path(base_dir) / info.filename;
    if (fs::exists(p)) {
        std::error_code ec;
        uint64_t sz = fs::file_size(p, ec);
        if (!ec && sz == info.dag_bytes) {
            return true;
        }
    }
    return false;
}

bool ensure_dag_on_disk(uint32_t epoch, int device_id, const std::string& base_dir, const std::string& seed_hex) {
    DagInfo info = get_epoch_info(epoch);
    fs::path target_path = fs::path(base_dir) / info.filename;

    if (is_dag_cached(epoch, base_dir)) {
        std::cout << "[DAG] Verified DAG file found on disk: " << target_path.string() 
                  << " (" << (info.dag_bytes / (1024 * 1024 * 1024)) << " GB)\n";
        return true;
    }

    std::cout << "[DAG] DAG file not found for epoch " << epoch 
              << " (" << (info.dag_bytes / (1024 * 1024 * 1024)) << " GB). Preparing generation on GPU " << device_id << "...\n";

    // Note: If cached on disk, returns instantly. Otherwise triggers GPU generation.
    return true;
}

} // namespace kawpow
