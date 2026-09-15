# KAWPOW Distributed Mining System (C++20)

High-performance C++20 implementation of the distributed KAWPOW (Ravencoin) mining infrastructure.

## Architecture

The distributed mining system separates responsibilities into two distinct, high-throughput components:

```
                  +-----------------------------------+
                  |      Mining Pool (Stratum)        |
                  |     (e.g., 2miners:6060)          |
                  +-----------------+-----------------+
                                    | Stratum V1
                                    v
                  +-----------------+-----------------+
                  |      raven_mining_server          |
                  |  - Stratum pool coordinator       |
                  |  - Local DAG cache/verification   |
                  |  - HTTP DAG distribution (8080)   |
                  |  - Client coordinator (8088)      |
                  |  - Instant share dispatch queue   |
                  +-------+-------------------+-------+
                          |                   |
            HTTP DAG / Job|                   |HTTP DAG / Job
           (TCP / JSON)   v                   v
              +-----------+---+           +---+-----------+
              | mining_client |           | mining_client |
              |  Rig 1 (GPUs) |           |  Rig 2 (GPUs) |
              +---------------+           +---------------+
```

1. **`raven_mining_server`**
   - Connects to Ravencoin Stratum pools (e.g., `rvn.2miners.com:6060`).
   - Serves epoch DAG caches over HTTP (port `8080`) to all connected client machines.
   - Maintains a low-latency TCP Coordinator (port `8088`) for client rig discovery, real-time job broadcasts, and epoch shifts.
   - Collects verified shares via a thread-safe FIFO queue and submits them instantaneously to the pool.

2. **`raven_mining_client`**
   - Auto-discovers and queries the server for DAG caches via HTTP, streaming them directly into local disk cache.
   - Multi-GPU execution: spawns independent mining threads per GPU index with non-overlapping search spaces (`[gpu_idx * 2^48, (gpu_idx + 1) * 2^48)`).
   - Real-time hardware telemetry integration via NVML (`lib-fg-cuda`): monitors temperatures, power consumption (W), fan speeds, and memory controller loads.
   - Reports live hashrate to the coordinator and submits valid shares immediately.

---

## Building

### Requirements
- **CMake**: >= 3.20
- **C++ Compiler**: Modern C++20 compiler:
  - MSVC (Visual Studio 2022 v19.30+)
  - GCC 11+ or Clang 13+
- **Windows**: `ws2_32.lib` (automatically linked)
- **CUDA Toolkit** (optional / runtime dynamically loaded via NVML `nvml.dll` / `libnvidia-ml.so`)

### Build Steps

```bash
cd distributed-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

The compiled binaries will be placed in:
- `build/Release/raven_mining_server.exe`
- `build/Release/raven_mining_client.exe`

---

## Usage

### 1. Launching the Coordinator Server

```bash
./raven_mining_server --pool rvn.2miners.com:6060 --wallet RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi --worker rig_server --http-port 8080 --client-port 8088
```

#### Server Options:
- `--pool <host:port>`: Stratum pool endpoint (default: `rvn.2miners.com:6060`)
- `--wallet <address>`: Mining wallet address
- `--worker <name>`: Coordinator worker rig name (default: `rig_server`)
- `--http-port <port>`: HTTP port for DAG binary file serving (default: `8080`)
- `--client-port <port>`: TCP port for mining client coordination (default: `8088`)
- `--gpu <id>`: GPU ID for local DAG generation (default: `0`)

---

### 2. Launching Worker Mining Clients

```bash
# Single-GPU setup (GPU 0)
./raven_mining_client --server 127.0.0.1:8088 --http-server http://127.0.0.1:8080 --worker client_rig_1 --gpus 0

# Multi-GPU setup (GPUs 0, 1, 2)
./raven_mining_client --server 192.168.1.100:8088 --http-server http://192.168.1.100:8080 --worker multi_gpu_rig --gpus 0,1,2 --batch-size 524288
```

#### Client Options:
- `--server <host:port>`: Address of the coordinator server (default: `127.0.0.1:8088`)
- `--http-server <url>`: URL of the HTTP DAG distribution server (default: `http://127.0.0.1:8080`)
- `--worker <name>`: Rig identifier (default: `client_rig_1`)
- `--batch-size <size>`: Batch size per search iteration (default: `524288`)
- `--gpus <id1,id2,...>`: Comma-separated GPU indices (default: `0`)
