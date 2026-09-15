#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>

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

// Resolves path of cached DAG file if it exists and matches exact byte size.
// Searches base_dir, current directory, parent (..), and grandparent (../..).
std::string find_cached_dag_path(uint32_t epoch, const std::string& base_dir = ".");

// Checks if DAG file exists on disk and is verified by byte size
bool is_dag_cached(uint32_t epoch, const std::string& base_dir = ".");

// Ensures verified DAG exists on disk synchronously; if missing, triggers generation
bool ensure_dag_on_disk(uint32_t epoch, int device_id = 0, const std::string& base_dir = ".", const std::string& seed_hex = "");

// Thread-safe DAG generation manager that handles epoch transitions, background synthesis,
// and ensures ongoing generation is NOT restarted on subsequent jobs for the same epoch.
class DagGenerator {
public:
    DagGenerator();
    ~DagGenerator();

    // Called on new pool job arrival.
    // If DAG is already generating for the same epoch, continues uninterrupted without restart.
    // Only restarts if the new job has a different epoch.
    void on_pool_job(uint32_t epoch, int device_id = 0, const std::string& base_dir = ".",
                     std::function<void(uint32_t)> on_ready_callback = nullptr);

    bool is_ready(uint32_t epoch) const;
    bool is_generating() const;
    uint32_t get_generating_epoch() const;
    std::string get_dag_path(uint32_t epoch) const;
    void stop();

private:
    mutable std::mutex mutex_;
    std::atomic<bool> generating_{false};
    std::atomic<uint32_t> generating_epoch_{UINT32_MAX};
    std::atomic<bool> cancel_flag_{false};
    std::thread worker_thread_;
    std::string resolved_path_;
    std::string base_dir_ = ".";
    int device_id_ = 0;
    std::function<void(uint32_t)> on_ready_cb_;

    void generate_worker(uint32_t epoch, int device_id, std::string base_dir);
};

} // namespace kawpow
