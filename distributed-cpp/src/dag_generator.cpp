#include "dag_generator.hpp"
#include "engine_core.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <vector>
#include <cstdlib>

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

std::string find_cached_dag_path(uint32_t epoch, const std::string& base_dir) {
    DagInfo info = get_epoch_info(epoch);
    std::vector<fs::path> candidates = {
        fs::path(base_dir) / info.filename,
        fs::path(".") / info.filename,
        fs::path("..") / info.filename,
        fs::path("../..") / info.filename
    };

    for (const auto& p : candidates) {
        std::error_code ec;
        if (fs::exists(p, ec)) {
            uint64_t sz = fs::file_size(p, ec);
            if (!ec && sz == info.dag_bytes) {
                return p.lexically_normal().string();
            }
        }
    }
    return "";
}

bool is_dag_cached(uint32_t epoch, const std::string& base_dir) {
    return !find_cached_dag_path(epoch, base_dir).empty();
}

bool ensure_dag_on_disk(uint32_t epoch, int device_id, const std::string& base_dir, const std::string& /*seed_hex*/) {
    DagInfo info = get_epoch_info(epoch);
    std::string existing = find_cached_dag_path(epoch, base_dir);
    if (!existing.empty()) {
        std::cout << "[DAG] Verified DAG file found on disk: " << existing 
                  << " (" << (info.dag_bytes / (1024 * 1024 * 1024)) << " GB)\n";
        return true;
    }

    std::cout << "[DAG] DAG file not found for epoch " << epoch 
              << " (" << (info.dag_bytes / (1024 * 1024 * 1024)) << " GB). Preparing generation on GPU " << device_id << "...\n";
    return false;
}

// -------------------------------------------------------------
// DagGenerator Implementation
// -------------------------------------------------------------

DagGenerator::DagGenerator() = default;

DagGenerator::~DagGenerator() {
    stop();
}

void DagGenerator::stop() {
    cancel_flag_.store(true);
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    generating_.store(false);
    generating_epoch_.store(UINT32_MAX);
}

bool DagGenerator::is_ready(uint32_t epoch) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!resolved_path_.empty()) {
        std::error_code ec;
        if (fs::exists(resolved_path_, ec)) {
            auto info = get_epoch_info(epoch);
            if (fs::file_size(resolved_path_, ec) == info.dag_bytes) {
                return true;
            }
        }
    }
    return !find_cached_dag_path(epoch, base_dir_).empty();
}

bool DagGenerator::is_generating() const {
    return generating_.load();
}

uint32_t DagGenerator::get_generating_epoch() const {
    return generating_epoch_.load();
}

std::string DagGenerator::get_dag_path(uint32_t epoch) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!resolved_path_.empty()) {
        return resolved_path_;
    }
    return find_cached_dag_path(epoch, base_dir_);
}

void DagGenerator::on_pool_job(uint32_t epoch, int device_id, const std::string& base_dir,
                              std::function<void(uint32_t)> on_ready_callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    base_dir_ = base_dir;
    device_id_ = device_id;
    if (on_ready_callback) {
        on_ready_cb_ = on_ready_callback;
    }

    // 1. If DAG is already cached on disk, update resolved path and we're done
    std::string existing = find_cached_dag_path(epoch, base_dir);
    if (!existing.empty()) {
        resolved_path_ = existing;
        if (generating_.load()) {
            cancel_flag_.store(true);
            if (worker_thread_.joinable()) worker_thread_.detach();
            generating_.store(false);
            generating_epoch_.store(UINT32_MAX);
        }
        return;
    }

    // 2. DAG file is NOT on disk yet.
    // Check if generation is already running:
    if (generating_.load()) {
        if (generating_epoch_.load() == epoch) {
            // CRITICAL: Ongoing generation is for the SAME epoch.
            // Do NOT restart! Pool sends new jobs every few seconds, ignore repeated triggers.
            return;
        } else {
            // CRITICAL: Epoch changed while generation is in progress.
            // Restart generation ONLY because the epoch is different.
            std::cout << "[DAG] *** Epoch changed from " << generating_epoch_.load()
                      << " to " << epoch << " during generation. Cancelling and restarting... ***\n";
            cancel_flag_.store(true);
            if (worker_thread_.joinable()) {
                worker_thread_.detach();
            }
            generating_.store(false);
            generating_epoch_.store(UINT32_MAX);
        }
    }

    // 3. Launch background generation for the new epoch
    cancel_flag_.store(false);
    generating_.store(true);
    generating_epoch_.store(epoch);

    auto info = get_epoch_info(epoch);
    std::cout << "[DAG] DAG file not found for epoch " << epoch 
              << " (" << (info.dag_bytes / (1024 * 1024 * 1024)) 
              << " GB). Starting background generation on GPU " << device_id << "...\n";

    worker_thread_ = std::thread(&DagGenerator::generate_worker, this, epoch, device_id, base_dir);
}

void DagGenerator::generate_worker(uint32_t epoch, int device_id, std::string base_dir) {
    auto info = get_epoch_info(epoch);
    fs::path final_path = fs::path(base_dir) / info.filename;
    fs::path tmp_path = fs::path(base_dir) / (info.filename + ".tmp");

    auto t0 = std::chrono::steady_clock::now();
    std::cout << "[DAG] Generating epoch " << epoch << " DAG (" 
              << (info.dag_bytes / (1024 * 1024 * 1024)) << " GB) on GPU " << device_id << "...\n";

    // Build DAG synthesis command using Python/PyCUDA engine if available
    std::string py_cmd = "python -c \""
        "import sys; "
        "from distributed.kawpow_server import MiningCoordinatorServer; "
        "srv = MiningCoordinatorServer(device_id=" + std::to_string(device_id) + "); "
        "srv.ensure_dag_on_disk(" + std::to_string(epoch) + ")\"";

    int ret = std::system(py_cmd.c_str());

    if (cancel_flag_.load()) {
        std::cout << "[DAG] Generation for epoch " << epoch << " was cancelled.\n";
        std::error_code ec;
        fs::remove(tmp_path, ec);
        generating_.store(false);
        generating_epoch_.store(UINT32_MAX);
        return;
    }

    // Re-check disk cache across candidate locations
    std::string found = find_cached_dag_path(epoch, base_dir);
    if (!found.empty()) {
        auto dt = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "[DAG] Epoch " << epoch << " DAG ready on disk: " << found 
                  << " (completed in " << dt << "s)\n";
        {
            std::lock_guard<std::mutex> lock(mutex_);
            resolved_path_ = found;
        }
        generating_.store(false);
        generating_epoch_.store(UINT32_MAX);

        if (on_ready_cb_) {
            on_ready_cb_(epoch);
        }
    } else {
        std::cerr << "[DAG] Warning: DAG generation process exited with code " << ret 
                  << " but verified file was not found.\n";
        generating_.store(false);
        generating_epoch_.store(UINT32_MAX);
    }
}

} // namespace kawpow
