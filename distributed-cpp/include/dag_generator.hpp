#pragma once

#include <string>
#include <cstdint>
#include <vector>

namespace kawpow {

struct DagInfo {
    uint32_t epoch = 0;
    uint64_t dag_bytes = 0;
    uint32_t num_cache_items = 0;
    uint32_t num_dag_items_2048 = 0;
    std::string filename;
    bool ready = false;
};

// Computes epoch sizing metadata
DagInfo get_epoch_info(uint32_t epoch);

// Checks if DAG file exists on disk and is verified by byte size
bool is_dag_cached(uint32_t epoch, const std::string& base_dir = ".");

// Ensures verified DAG exists on disk; if missing, synthesizes on GPU
bool ensure_dag_on_disk(uint32_t epoch, int device_id = 0, const std::string& base_dir = ".", const std::string& seed_hex = "");

} // namespace kawpow
