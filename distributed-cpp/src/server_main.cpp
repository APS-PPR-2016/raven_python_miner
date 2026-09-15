#include "engine_core.hpp"
#include "dag_generator.hpp"
#include "network.hpp"
#include "simple_json.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <map>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

struct ShareData {
    uint32_t client_id;
    std::string worker_name;
    std::string job_id;
    std::string nonce;
    std::string header;
    std::string mix_hash;
};

struct ClientSession {
    uint32_t id;
    std::string worker_name;
    int gpus = 1;
    double hashrate = 0.0;
    std::shared_ptr<kawpow::net::TcpSocket> sock;
};

class MiningServer {
public:
    std::string pool_host = "rvn.2miners.com";
    int pool_port = 6060;
    std::string wallet = "RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi";
    std::string worker = "kawpow_server";
    int http_port = 8080;
    int client_port = 8088;
    int device_id = 0;

    std::atomic<bool> running{true};
    std::atomic<uint32_t> current_epoch{604};
    std::atomic<uint64_t> current_height{4536814};
    std::string current_job;
    std::string current_blob;
    std::string current_target = "00000000ffff0000000000000000000000000000000000000000000000000000";

    std::mutex job_mutex;
    std::mutex clients_mutex;
    std::map<uint32_t, ClientSession> clients;
    uint32_t next_client_id = 1;

    std::mutex share_mutex;
    std::condition_variable share_cv;
    std::queue<ShareData> share_queue;

    kawpow::net::TcpSocket pool_sock;
    std::mutex pool_sock_mutex;

    void broadcast_job() {
        simple_json::Value job_msg(std::map<std::string, simple_json::Value>{
            {"type", "job"},
            {"jid", current_job},
            {"job_id", current_job},
            {"bl", current_blob},
            {"blob", current_blob},
            {"t", current_target},
            {"target", current_target},
            {"h", static_cast<int64_t>(current_height.load())},
            {"height", static_cast<int64_t>(current_height.load())},
            {"ep", static_cast<int64_t>(current_epoch.load())},
            {"epoch", static_cast<int64_t>(current_epoch.load())}
        });

        std::string line = job_msg.serialize() + "\n";
        std::lock_guard<std::mutex> lock(clients_mutex);
        for (auto& [id, c] : clients) {
            if (c.sock && c.sock->is_valid()) {
                c.sock->send_string(line);
            }
        }
    }

    void handle_http_client(kawpow::net::TcpSocket client) {
        std::string line;
        if (!client.readline(line)) return;

        std::istringstream iss(line);
        std::string method, path, proto;
        iss >> method >> path >> proto;

        // Read remaining headers
        while (client.readline(line)) {
            if (line.empty() || line == "\r") break;
        }

        if (path == "/dag/info") {
            auto info = kawpow::get_epoch_info(current_epoch.load());
            bool ready = kawpow::is_dag_cached(info.epoch);
            simple_json::Value resp(std::map<std::string, simple_json::Value>{
                {"epoch", static_cast<int64_t>(info.epoch)},
                {"dag_bytes", static_cast<int64_t>(info.dag_bytes)},
                {"filename", info.filename},
                {"ready", ready}
            });
            std::string body = resp.serialize();
            std::string header = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                                 std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
            client.send_string(header + body);
        } else if (path.rfind("/dag/download", 0) == 0) {
            auto info = kawpow::get_epoch_info(current_epoch.load());
            if (!kawpow::is_dag_cached(info.epoch)) {
                std::string err = "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 5\r\nContent-Length: 26\r\n\r\nDAG generating, retry in 5s";
                client.send_string(err);
            } else {
                std::ifstream f(info.filename, std::ios::binary);
                if (!f) {
                    std::string not_found = "HTTP/1.1 404 Not Found\r\n\r\nDAG file missing";
                    client.send_string(not_found);
                    return;
                }
                std::string headers = "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: " +
                                      std::to_string(info.dag_bytes) + "\r\nContent-Disposition: attachment; filename=" +
                                      info.filename + "\r\nConnection: close\r\n\r\n";
                client.send_string(headers);

                // Stream in 8MB chunks
                const size_t chunk_size = 8 * 1024 * 1024;
                std::vector<char> buf(chunk_size);
                while (f && client.is_valid()) {
                    f.read(buf.data(), chunk_size);
                    std::streamsize bytes_read = f.gcount();
                    if (bytes_read <= 0) break;
                    if (!client.send(buf.data(), static_cast<size_t>(bytes_read))) break;
                }
            }
        } else {
            std::string nf = "HTTP/1.1 404 Not Found\r\n\r\nInvalid endpoint";
            client.send_string(nf);
        }
    }

    void run_http_server() {
        kawpow::net::TcpServer srv;
        if (!srv.bind_and_listen(http_port)) {
            std::cerr << "[Server] Failed to bind HTTP server on port " << http_port << "\n";
            return;
        }
        std::cout << "[Server] HTTP DAG Server listening on http://0.0.0.0:" << http_port << "\n";

        while (running) {
            auto client = srv.accept();
            if (!client.is_valid()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            std::thread([this, c = std::move(client)]() mutable {
                handle_http_client(std::move(c));
            }).detach();
        }
    }

    void handle_mining_client(std::shared_ptr<kawpow::net::TcpSocket> sock, uint32_t client_id) {
        uint64_t nonce_prefix = static_cast<uint64_t>(client_id) << 40;
        std::cout << "[Server] Client " << client_id << " connected. Nonce prefix: 0x" << std::hex << nonce_prefix << std::dec << "\n";

        // Send init message
        simple_json::Value init_msg(std::map<std::string, simple_json::Value>{
            {"type", "init"},
            {"c_id", static_cast<int64_t>(client_id)},
            {"client_id", static_cast<int64_t>(client_id)},
            {"np", static_cast<int64_t>(nonce_prefix)},
            {"nonce_prefix", static_cast<int64_t>(nonce_prefix)},
            {"ep", static_cast<int64_t>(current_epoch.load())},
            {"epoch", static_cast<int64_t>(current_epoch.load())},
            {"dh", "http://localhost:" + std::to_string(http_port) + "/dag/download"}
        });
        sock->send_string(init_msg.serialize() + "\n");

        // Send active job if available
        {
            std::lock_guard<std::mutex> lock(job_mutex);
            if (!current_job.empty()) {
                simple_json::Value job_msg(std::map<std::string, simple_json::Value>{
                    {"type", "job"},
                    {"jid", current_job},
                    {"job_id", current_job},
                    {"bl", current_blob},
                    {"blob", current_blob},
                    {"t", current_target},
                    {"target", current_target},
                    {"h", static_cast<int64_t>(current_height.load())},
                    {"height", static_cast<int64_t>(current_height.load())},
                    {"ep", static_cast<int64_t>(current_epoch.load())},
                    {"epoch", static_cast<int64_t>(current_epoch.load())}
                });
                sock->send_string(job_msg.serialize() + "\n");
            }
        }

        std::string line;
        while (running && sock->readline(line)) {
            if (line.empty()) continue;
            auto msg = simple_json::parse(line);
            std::string type = msg["type"].as_string();

            if (type == "register") {
                std::lock_guard<std::mutex> lock(clients_mutex);
                if (clients.count(client_id)) {
                    clients[client_id].worker_name = msg["worker_name"].as_string_or("worker_" + std::to_string(client_id));
                    clients[client_id].gpus = static_cast<int>(msg["gpus"].as_int64(1));
                    std::cout << "[Server] Client " << client_id << " registered as '" << clients[client_id].worker_name
                              << "' (" << clients[client_id].gpus << " GPU(s))\n";
                }
            } else if (type == "share" || type == "sh") {
                ShareData s;
                s.client_id = client_id;
                {
                    std::lock_guard<std::mutex> lock(clients_mutex);
                    s.worker_name = clients.count(client_id) ? clients[client_id].worker_name : "worker_" + std::to_string(client_id);
                }
                s.job_id = msg.contains("jid") ? msg["jid"].as_string() : msg["job_id"].as_string();
                s.nonce = msg.contains("n") ? msg["n"].as_string() : msg["nonce"].as_string();
                s.header = msg.contains("h") ? msg["h"].as_string() : msg["header"].as_string();
                s.mix_hash = msg.contains("m") ? msg["m"].as_string() : msg["mix_hash"].as_string();

                {
                    std::lock_guard<std::mutex> lock(share_mutex);
                    share_queue.push(std::move(s));
                }
                share_cv.notify_one();
            } else if (type == "hashrate" || type == "hr") {
                double hr = msg.contains("hr") ? msg["hr"].as_double() : msg["hashrate"].as_double();
                std::lock_guard<std::mutex> lock(clients_mutex);
                if (clients.count(client_id)) clients[client_id].hashrate = hr;
            }
        }

        std::cout << "[Server] Client " << client_id << " disconnected.\n";
        {
            std::lock_guard<std::mutex> lock(clients_mutex);
            clients.erase(client_id);
        }
    }

    void run_coordinator_server() {
        kawpow::net::TcpServer srv;
        if (!srv.bind_and_listen(client_port)) {
            std::cerr << "[Server] Failed to bind Client Coordinator on port " << client_port << "\n";
            return;
        }
        std::cout << "[Server] Mining Client Coordinator listening on port " << client_port << "\n";

        while (running) {
            auto client_sock = srv.accept();
            if (!client_sock.is_valid()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            uint32_t cid = next_client_id++;
            auto s_ptr = std::make_shared<kawpow::net::TcpSocket>(std::move(client_sock));
            {
                std::lock_guard<std::mutex> lock(clients_mutex);
                clients[cid] = ClientSession{cid, "worker_" + std::to_string(cid), 1, 0.0, s_ptr};
            }

            std::thread([this, s_ptr, cid]() {
                handle_mining_client(s_ptr, cid);
            }).detach();
        }
    }

    void run_share_dispatcher() {
        while (running) {
            ShareData s;
            {
                std::unique_lock<std::mutex> lock(share_mutex);
                share_cv.wait(lock, [this] { return !share_queue.empty() || !running; });
                if (!running) break;
                s = std::move(share_queue.front());
                share_queue.pop();
            }

            // Dispatch to Stratum pool
            std::string user = wallet + "." + s.worker_name;
            simple_json::Value submit_msg(std::map<std::string, simple_json::Value>{
                {"id", 4},
                {"method", "mining.submit"},
                {"params", std::vector<simple_json::Value>{user, s.job_id, s.nonce, s.header, s.mix_hash}}
            });

            std::lock_guard<std::mutex> lock(pool_sock_mutex);
            if (pool_sock.is_valid()) {
                pool_sock.send_string(submit_msg.serialize() + "\n");
                std::cout << "[Server] >>> Submitted share from " << s.worker_name 
                          << ": nonce=" << s.nonce << ", mix=" << s.mix_hash.substr(0, 16) 
                          << "... for job=" << s.job_id << "\n";
            }
        }
    }

    void connect_stratum_pool() {
        while (running) {
            std::cout << "[Server] Connecting to Stratum pool at " << pool_host << ":" << pool_port << "...\n";
            {
                std::lock_guard<std::mutex> lock(pool_sock_mutex);
                if (!pool_sock.connect(pool_host, pool_port)) {
                    std::cerr << "[Server] Pool connection failed. Reconnecting in 5s...\n";
                }
            }

            if (!pool_sock.is_valid()) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                continue;
            }

            std::string user = wallet + "." + worker;

            // 1. Subscribe
            simple_json::Value sub_msg(std::map<std::string, simple_json::Value>{
                {"id", 1},
                {"method", "mining.subscribe"},
                {"params", std::vector<simple_json::Value>{user, "x"}}
            });
            pool_sock.send_string(sub_msg.serialize() + "\n");
            std::string sub_resp;
            if (!pool_sock.readline(sub_resp)) continue;

            // 2. Authorize
            simple_json::Value auth_msg(std::map<std::string, simple_json::Value>{
                {"id", 2},
                {"method", "mining.authorize"},
                {"params", std::vector<simple_json::Value>{user, "x"}}
            });
            pool_sock.send_string(auth_msg.serialize() + "\n");
            std::string auth_resp;
            if (!pool_sock.readline(auth_resp)) continue;

            std::cout << "[Server] Connected & authorized to Stratum pool at " << pool_host << ":" << pool_port << "\n";

            // 3. Pool message listener loop
            std::string line;
            while (running && pool_sock.readline(line)) {
                if (line.empty()) continue;
                auto msg = simple_json::parse(line);

                if (msg.contains("method")) {
                    std::string method = msg["method"].as_string();
                    if (method == "mining.notify") {
                        auto params = msg["params"];
                        if (params.arr_val.size() >= 4) {
                            std::string jid = params[0].as_string();
                            std::string bl = params[1].as_string();
                            std::string targ = params[3].as_string();
                            uint64_t h = params.arr_val.size() >= 6 ? params[5].as_uint64() : 0;
                            uint32_t ep = static_cast<uint32_t>(h / 7500);

                            {
                                std::lock_guard<std::mutex> lock(job_mutex);
                                current_job = jid;
                                current_blob = bl;
                                current_target = targ;
                                current_height = h;
                                current_epoch = ep;
                            }

                            std::cout << "[Pool] Job: id=" << jid << ", height=" << h << ", epoch=" << ep << "\n";
                            kawpow::ensure_dag_on_disk(ep, device_id);
                            broadcast_job();
                        }
                    }
                } else if (msg.contains("id") && msg["id"].as_int64() == 4) {
                    if (msg.contains("result") && msg["result"].as_bool()) {
                        std::cout << "[Pool] >>> Share ACCEPTED! <<<\n";
                    } else {
                        std::cout << "[Pool] Share REJECTED: " << msg["error"].serialize() << "\n";
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
};

int main(int argc, char* argv[]) {
    MiningServer srv;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: raven_mining_server [options]\n\n"
                      << "Options:\n"
                      << "  --pool <host:port>     Stratum pool (default: rvn.2miners.com:6060)\n"
                      << "  --wallet <address>     Wallet address\n"
                      << "  --worker <name>        Worker rig name (default: rig_server)\n"
                      << "  --http-port <port>     HTTP port for DAG file serving (default: 8080)\n"
                      << "  --client-port <port>   TCP port for mining client coordinator (default: 8088)\n"
                      << "  --gpu <id>             GPU device ID for local DAG generation (default: 0)\n"
                      << "  --help, -h             Show this help message\n";
            return 0;
        } else if (arg == "--pool" && i + 1 < argc) {
            std::string p = argv[++i];
            size_t colon = p.find(':');
            if (colon != std::string::npos) {
                srv.pool_host = p.substr(0, colon);
                srv.pool_port = std::stoi(p.substr(colon + 1));
            }
        } else if (arg == "--wallet" && i + 1 < argc) {
            srv.wallet = argv[++i];
        } else if (arg == "--worker" && i + 1 < argc) {
            srv.worker = argv[++i];
        } else if (arg == "--http-port" && i + 1 < argc) {
            srv.http_port = std::stoi(argv[++i]);
        } else if (arg == "--client-port" && i + 1 < argc) {
            srv.client_port = std::stoi(argv[++i]);
        } else if (arg == "--gpu" && i + 1 < argc) {
            srv.device_id = std::stoi(argv[++i]);
        }
    }

    std::cout << "=== KAWPOW C++20 Distributed Mining Server ===\n";
    std::cout << "Pool: " << srv.pool_host << ":" << srv.pool_port << " | Wallet: " << srv.wallet << "\n";

    std::thread t_http([&]() { srv.run_http_server(); });
    std::thread t_coord([&]() { srv.run_coordinator_server(); });
    std::thread t_shares([&]() { srv.run_share_dispatcher(); });

    srv.connect_stratum_pool();

    srv.running = false;
    if (t_http.joinable()) t_http.join();
    if (t_coord.joinable()) t_coord.join();
    if (t_shares.joinable()) t_shares.join();

    return 0;
}
