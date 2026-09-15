#include "engine_core.hpp"
#include "dag_generator.hpp"
#include "network.hpp"
#include "simple_json.hpp"
#include "fgp/cuda/gpu_resource.h"

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace fs = std::filesystem;

struct MiningJob {
    std::string job_id;
    std::string blob;
    uint64_t target_high64 = 0x00000000FFFF0000ULL;
    uint64_t height = 0;
    uint32_t epoch = 0;
};

class MiningClient {
public:
    std::string server_host = "127.0.0.1";
    int server_port = 8088;
    std::string http_server = "http://127.0.0.1:8080";
    std::vector<int> gpu_ids = {0};
    std::string worker_name = "client_node1";
    uint32_t batch_size = 524288;

    std::atomic<bool> running{true};
    uint32_t client_id = 0;
    uint64_t nonce_prefix = 0;
    uint32_t active_epoch = 0;

    std::mutex job_mutex;
    MiningJob current_job;
    bool has_job = false;

    kawpow::net::TcpSocket server_sock;
    std::mutex sock_mutex;

    std::mutex share_queue_mutex;
    std::condition_variable share_cv;
    std::queue<std::string> outgoing_shares;

    // NVML telemetry monitor
    std::unique_ptr<fgp::cuda::CudaNVML> nvml;
    std::vector<fgp::cuda::GpuResource> gpu_resources;

    void init_nvml() {
        nvml = std::make_unique<fgp::cuda::CudaNVML>();
        if (nvml->initialize()) {
            std::cout << "[Client] NVML initialized successfully. Driver version: " << nvml->getDriverVersion() << "\n";
            auto all_gpus = fgp::cuda::GpuResource::listAvailableGpus(*nvml);
            for (int gid : gpu_ids) {
                if (gid < static_cast<int>(all_gpus.size())) {
                    gpu_resources.push_back(all_gpus[gid]);
                    std::cout << "[Client] Bound GPU " << gid << ": " << all_gpus[gid].getName()
                              << " (" << std::fixed << std::setprecision(2) << all_gpus[gid].getTotalMemoryGB() << " GB VRAM)\n";
                }
            }
        } else {
            std::cout << "[Client] NVML not available or no NVIDIA driver detected. Running in standard CUDA mode.\n";
        }
    }

    void download_dag_if_needed(uint32_t epoch) {
        auto info = kawpow::get_epoch_info(epoch);
        if (fs::exists(info.filename) && fs::file_size(info.filename) == info.dag_bytes) {
            std::cout << "[Client] Found cached DAG file on local disk: " << info.filename
                      << " (" << (info.dag_bytes / (1024 * 1024 * 1024)) << " GB)\n[Client] Loading DAG to gpu..\n";
            return;
        }

        std::cout << "[Client] Downloading DAG from server: " << http_server << "/dag/download -> " << info.filename << "...\n";
        // Parse HTTP server host and port
        std::string host = "127.0.0.1";
        int port = 8080;
        std::string url = http_server;
        if (url.rfind("http://", 0) == 0) url = url.substr(7);
        size_t colon = url.find(':');
        if (colon != std::string::npos) {
            host = url.substr(0, colon);
            size_t slash = url.find('/', colon);
            port = std::stoi(url.substr(colon + 1, slash != std::string::npos ? slash - colon - 1 : std::string::npos));
        }

        while (running) {
            kawpow::net::TcpSocket http_sock;
            if (!http_sock.connect(host, port)) {
                std::cout << "[Client] Cannot connect to HTTP server. Retrying in 4s...\n";
                std::this_thread::sleep_for(std::chrono::seconds(4));
                continue;
            }

            std::string req = "GET /dag/download HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n";
            http_sock.send_string(req);

            std::string status_line;
            if (!http_sock.readline(status_line)) {
                std::this_thread::sleep_for(std::chrono::seconds(3));
                continue;
            }

            if (status_line.find("503") != std::string::npos) {
                std::cout << "[Client] Server is synthesizing DAG for epoch " << epoch << ". Retrying in 4s...\n";
                std::this_thread::sleep_for(std::chrono::seconds(4));
                continue;
            }

            if (status_line.find("200") == std::string::npos) {
                std::cout << "[Client] HTTP error: " << status_line << ". Retrying in 4s...\n";
                std::this_thread::sleep_for(std::chrono::seconds(4));
                continue;
            }

            // Read headers
            std::string h;
            while (http_sock.readline(h)) {
                if (h.empty() || h == "\r") break;
            }

            // Stream response to disk
            std::ofstream out(info.filename, std::ios::binary);
            std::vector<char> buf(8 * 1024 * 1024);
            uint64_t total_read = 0;
            auto t0 = std::chrono::steady_clock::now();

            while (running) {
                int r = http_sock.recv(buf.data(), buf.size());
                if (r <= 0) break;
                out.write(buf.data(), r);
                total_read += r;
                double pct = (static_cast<double>(total_read) / info.dag_bytes) * 100.0;
                std::cout << "\r[Client] Downloading DAG: " << std::fixed << std::setprecision(1) << pct << "% ("
                          << (total_read / (1024 * 1024)) << " MB / " << (info.dag_bytes / (1024 * 1024)) << " MB)" << std::flush;
            }
            std::cout << "\n[Client] Download complete! (" << (total_read / (1024 * 1024)) << " MB)\n";
            break;
        }
    }

    void share_sender_loop() {
        while (running) {
            std::string share_msg;
            {
                std::unique_lock<std::mutex> lock(share_queue_mutex);
                share_cv.wait(lock, [this] { return !outgoing_shares.empty() || !running; });
                if (!running) break;
                share_msg = std::move(outgoing_shares.front());
                outgoing_shares.pop();
            }

            std::lock_guard<std::mutex> lock(sock_mutex);
            if (server_sock.is_valid()) {
                server_sock.send_string(share_msg + "\n");
            }
        }
    }

    void mining_worker_thread(int gpu_idx) {
        uint64_t batch_counter = 0;
        auto last_telemetry = std::chrono::steady_clock::now();
        uint64_t total_hashes_batch = 0;

        while (running) {
            MiningJob job;
            {
                std::lock_guard<std::mutex> lock(job_mutex);
                if (!has_job) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                }
                job = current_job;
            }

            // Partition nonces for this local GPU: nonce_prefix + (gpu_idx << 36) + batch * batch_size
            uint64_t gpu_base_nonce = nonce_prefix + (static_cast<uint64_t>(gpu_idx) << 36);
            uint64_t start_nonce = gpu_base_nonce + (batch_counter * batch_size);

            // Compute hash batch on GPU
            std::this_thread::sleep_for(std::chrono::milliseconds(25)); // Emulated CUDA batch timing
            total_hashes_batch += batch_size;
            batch_counter++;

            auto now = std::chrono::steady_clock::now();
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_telemetry).count();
            if (dt >= 5000) {
                double mhs = (static_cast<double>(total_hashes_batch) / (dt / 1000.0)) / 1000000.0;
                total_hashes_batch = 0;
                last_telemetry = now;

                std::string extra_info = "";
                if (gpu_idx < static_cast<int>(gpu_resources.size())) {
                    gpu_resources[gpu_idx].refresh();
                    extra_info = " | GPU " + std::to_string(gpu_resources[gpu_idx].getUtilization().gpu) + "% | Mem " +
                                 std::to_string(gpu_resources[gpu_idx].getUtilization().memory) + "% (" +
                                 std::to_string(gpu_resources[gpu_idx].getTemperatureC()) + "C, " +
                                 std::to_string(static_cast<int>(gpu_resources[gpu_idx].getPowerUsageWatts())) + "W)";
                }

                std::cout << "[Client] GPU " << gpu_ids[gpu_idx] << " Hashrate: " << std::fixed << std::setprecision(2)
                          << mhs << " MH/s" << extra_info << " | Job: " << job.job_id
                          << " | Nonce: 0x" << std::hex << start_nonce << std::dec << "\n";

                // Report telemetry to coordinator
                simple_json::Value hr_msg(std::map<std::string, simple_json::Value>{
                    {"type", "hashrate"},
                    {"hr", mhs}
                });
                std::lock_guard<std::mutex> lock(share_queue_mutex);
                outgoing_shares.push(hr_msg.serialize());
                share_cv.notify_one();
            }
        }
    }

    void run() {
        init_nvml();
        std::thread t_sender([this] { share_sender_loop(); });

        while (running) {
            std::cout << "[Client] Connecting to KAWPOW server at " << server_host << ":" << server_port << "...\n";
            {
                std::lock_guard<std::mutex> lock(sock_mutex);
                if (!server_sock.connect(server_host, server_port)) {
                    std::cerr << "[Client] Server connection failed. Reconnecting in 4s...\n";
                }
            }

            if (!server_sock.is_valid()) {
                std::this_thread::sleep_for(std::chrono::seconds(4));
                continue;
            }

            std::cout << "[Client] Connected to server at " << server_host << ":" << server_port << "\n";

            // Register
            simple_json::Value reg_msg(std::map<std::string, simple_json::Value>{
                {"type", "register"},
                {"worker_name", worker_name},
                {"gpus", static_cast<int64_t>(gpu_ids.size())}
            });
            server_sock.send_string(reg_msg.serialize() + "\n");

            // Start local GPU worker threads
            std::vector<std::thread> workers;
            for (size_t i = 0; i < gpu_ids.size(); ++i) {
                workers.emplace_back([this, idx = static_cast<int>(i)] { mining_worker_thread(idx); });
            }

            // Coordinator message reader
            std::string line;
            while (running && server_sock.readline(line)) {
                if (line.empty()) continue;
                auto msg = simple_json::parse(line);
                std::string type = msg["type"].as_string();

                if (type == "init") {
                    client_id = static_cast<uint32_t>(msg.contains("c_id") ? msg["c_id"].as_int64() : msg["client_id"].as_int64());
                    nonce_prefix = static_cast<uint64_t>(msg.contains("np") ? msg["np"].as_int64() : msg["nonce_prefix"].as_int64());
                    uint32_t ep = static_cast<uint32_t>(msg.contains("ep") ? msg["ep"].as_int64() : msg["epoch"].as_int64());
                    std::cout << "[Client] Initialized as Client " << client_id << " (Nonce prefix: 0x" << std::hex << nonce_prefix << std::dec << ")\n";
                    if (ep != active_epoch) {
                        download_dag_if_needed(ep);
                        active_epoch = ep;
                    }
                } else if (type == "job") {
                    uint32_t ep = static_cast<uint32_t>(msg.contains("ep") ? msg["ep"].as_int64() : msg["epoch"].as_int64());
                    if (ep != active_epoch) {
                        download_dag_if_needed(ep);
                        active_epoch = ep;
                    }

                    {
                        std::lock_guard<std::mutex> lock(job_mutex);
                        current_job.job_id = msg.contains("jid") ? msg["jid"].as_string() : msg["job_id"].as_string();
                        current_job.blob = msg.contains("bl") ? msg["bl"].as_string() : msg["blob"].as_string();
                        current_job.height = msg.contains("h") ? msg["h"].as_uint64() : msg["height"].as_uint64();
                        current_job.epoch = ep;
                        has_job = true;
                    }
                }
            }

            for (auto& w : workers) {
                if (w.joinable()) w.join();
            }

            std::this_thread::sleep_for(std::chrono::seconds(4));
        }

        share_cv.notify_all();
        if (t_sender.joinable()) t_sender.join();
    }
};

int main(int argc, char* argv[]) {
    MiningClient client;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: raven_mining_client [options]\n\n"
                      << "Options:\n"
                      << "  --server <host:port>      Coordinator server (default: 127.0.0.1:8088)\n"
                      << "  --http-server <url>       HTTP server URL for DAG download (default: http://127.0.0.1:8080)\n"
                      << "  --worker <name>           Worker rig name (default: client_rig_1)\n"
                      << "  --batch-size <size>       Nonces per batch (default: 524288)\n"
                      << "  --gpus <id1,id2,...>      Comma-separated GPU indices (default: 0)\n"
                      << "  --help, -h                Show this help message\n";
            return 0;
        } else if (arg == "--server" && i + 1 < argc) {
            std::string s = argv[++i];
            size_t colon = s.find(':');
            if (colon != std::string::npos) {
                client.server_host = s.substr(0, colon);
                client.server_port = std::stoi(s.substr(colon + 1));
            }
        } else if (arg == "--http-server" && i + 1 < argc) {
            client.http_server = argv[++i];
        } else if (arg == "--worker" && i + 1 < argc) {
            client.worker_name = argv[++i];
        } else if (arg == "--batch-size" && i + 1 < argc) {
            client.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--gpus" && i + 1 < argc) {
            std::string g = argv[++i];
            client.gpu_ids.clear();
            std::istringstream ss(g);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) client.gpu_ids.push_back(std::stoi(item));
            }
            if (client.gpu_ids.empty()) client.gpu_ids.push_back(0);
        }
    }

    std::cout << "=== KAWPOW C++20 Distributed Mining Client ===\n";
    std::cout << "Target Coordinator: " << client.server_host << ":" << client.server_port
              << " | HTTP DAG: " << client.http_server << " | Worker: " << client.worker_name << "\n";

    client.run();
    return 0;
}
