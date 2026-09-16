# Raven Python Miner

<p align="center">
  <img src="https://raw.githubusercontent.com/RavenProject/Ravencoin/master/src/qt/res/icons/bitcoin.png" width="100" alt="Ravencoin Logo" />
</p>

<p align="center">
  <b>High-Performance Distributed KAWPOW (Ravencoin) GPU Miner with Stratum Coordinator, HTTP DAG Streaming, and Multi-GPU Support</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-green.svg" alt="CUDA Supported" />
  <img src="https://img.shields.io/badge/Algorithm-KAWPOW%20(ProgPOW)-orange.svg" alt="Algorithm KAWPOW" />
  <img src="https://img.shields.io/badge/Protocol-Stratum%20v1-yellow.svg" alt="Stratum v1" />
  <img src="https://img.shields.io/badge/Security-Native%20TLS%201.3-brightgreen.svg" alt="TLS 1.3" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License MIT" />
</p>

---

## 📖 Overview

**Raven Python Miner** is an asynchronous, distributed mining engine built specifically for the **Ravencoin (RVN) KAWPOW** proof-of-work algorithm. 

Designed for both single-machine miners and large-scale cloud GPU fleets (such as Vast.ai, RunPod, and Lambda Labs), this miner decouples mining pool coordination from GPU hash execution:

```
                  ┌───────────────────────────────┐
                  │    Stratum Pool (2Miners)     │
                  └───────────────┬───────────────┘
                                  │ Stratum v1 TCP
                                  ▼
                  ┌───────────────────────────────┐
                  │   KAWPOW Server Coordinator   │
                  │   - Pools connections         │
                  │   - Synthesizes / Caches DAG  │
                  │   - HTTP DAG Streaming (:8080)│
                  │   - TLS Job Dispatcher (:8088)│
                  └───────────────┬───────────────┘
                                  │ TLS 1.3 / TCP
       ┌──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│ GPU Worker 1 │           │ GPU Worker 2 │           │ GPU Worker N │
│ (RTX 3090)   │           │ (RTX 3060)   │           │ (RTX 4090)   │
└──────────────┘           └──────────────┘           └──────────────┘
```

---

## ✨ Key Features

- **Consensus-Identical KAWPOW Engine**: Implements the official Ravencoin ProgPOW specification, including Keccak-f[800], Keccak-f[1600], KISS99 PRNG, dynamic 64-round unrolled execution, and memory-hard DAG access.
- **Distributed Fleet Architecture**: Run one lightweight server coordinator that manages pool jobs, share validation, and dispatching, while dozens of remote GPU instances execute searches.
- **Centralized HTTP DAG Streaming**: The server synthesizes the multi-gigabyte DAG (5.73 GB at Epoch 605) once and serves it to client workers via high-speed HTTP streaming. Workers avoid redundant generation and start hashing within seconds.
- **Native TLS 1.3 Encryption**: Secure worker-to-server communications over public cloud IPs with automatic self-signed X.509 certificate generation.
- **Dynamic Epoch Transitions**: Automatically detects block height epoch rollovers, generates the new DAG, updates worker VRAM, and resumes mining without downtime or manual restarts.
- **Full Multi-GPU & Nonce Partitioning**: Spreads nonce search spaces automatically across all designated GPUs per worker rig.
- **Real-Time NVML Telemetry**: Live reporting of per-GPU hashrate, compute utilization, memory utilization, temperature, and power consumption.

---

## ⚡ Hashrate Performance Reference

Typical performance on calibrated KAWPOW settings:

| GPU Model | Architecture | VRAM | Hashrate | Power | Efficiency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA GeForce RTX 3060** | Ampere | 12 GB GDDR6 | **~21 – 23 MH/s** | ~125 W | ~0.18 MH/W |
| **NVIDIA Quadro RTX 6000** | Turing | 24 GB GDDR6 | **~38 – 42 MH/s** | ~220 W | ~0.19 MH/W |
| **NVIDIA GeForce RTX 3080** | Ampere | 10 GB GDDR6X | **~47 – 50 MH/s** | ~250 W | ~0.20 MH/W |
| **NVIDIA GeForce RTX 3090** | Ampere | 24 GB GDDR6X | **~58 – 62 MH/s** | ~300 W | ~0.20 MH/W |
| **NVIDIA GeForce RTX 4090** | Ada Lovelace | 24 GB GDDR6X | **~75 – 82 MH/s** | ~320 W | ~0.24 MH/W |

---

## 🚀 Quick Start

For detailed installation instructions on Linux, Windows, and Cloud GPU instances, see **[INSTALL.md](INSTALL.md)**.

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/APS-PPR-2016/raven_python_miner.git
cd raven_python_miner

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launching Single-Machine Mining (Local Server + GPU Worker)

#### Step 1: Start Coordinator in Terminal 1
```bash
python3 distributed/kawpow_server.py \
  --pool-host rvn.2miners.com \
  --pool-port 6060 \
  --wallet RNGyc94iUJnYWCcshAobk2xjUYw7EvLsq4 \
  --rig-name worker_1 \
  --port 8088 \
  --http-port 8080 \
  --no-tls
```

#### Step 2: Start GPU Miner in Terminal 2
```bash
python3 distributed/kawpow_client.py \
  --server-host 127.0.0.1 \
  --server-port 8088 \
  --http-url http://127.0.0.1:8080 \
  --gpus 0 \
  --worker-name worker_1_gpu0 \
  --no-tls
```

---

## ☁️ Distributed Cloud Mining (Vast.ai / RunPod / AWS)

When renting cloud GPUs with high bandwidth:

1. **Host Coordinator** on your master server or VPS:
   ```bash
   python3 distributed/kawpow_server.py \
     --pool-host rvn.2miners.com \
     --pool-port 6060 \
     --wallet YOUR_RVN_WALLET_ADDRESS \
     --rig-name cloud_rig \
     --port 8088 \
     --http-port 8080
   ```
   *(By default, TLS 1.3 is enabled with auto-generated certificates in `server.crt`).*

2. **Run GPU Clients** on any number of rented cloud nodes:
   ```bash
   python3 distributed/kawpow_client.py \
     --server-host <COORDINATOR_IP> \
     --server-port 8088 \
     --http-url http://<COORDINATOR_IP>:8080 \
     --gpus 0,1 \
     --worker-name vast_node_1
   ```

---

## ⚙️ CLI Reference

### Coordinator Server (`kawpow_server.py`)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--pool-host` | `rvn.2miners.com` | Stratum mining pool hostname |
| `--pool-port` | `6060` | Stratum pool port |
| `--wallet` | Required | Ravencoin payout address |
| `--rig-name` | `default_rig` | Rig identifier sent to pool |
| `--port` | `8088` | Client coordinator TCP/TLS port |
| `--http-port` | `8080` | DAG binary HTTP download port |
| `--no-tls` | `False` | Disables TLS encryption (plaintext TCP) |
| `--cache-dir` | Current Dir | Directory to store generated DAG cache files |

### Client Worker (`kawpow_client.py`)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--server-host` | `localhost` | Coordinator server IP or hostname |
| `--server-port` | `8088` | Coordinator server port |
| `--http-url` | `http://localhost:8080` | Coordinator HTTP DAG download URL |
| `--gpus` | `0` | Comma-separated GPU indices (e.g. `0,1,2`) |
| `--batch-size` | `524288` | Number of nonces searched per GPU batch |
| `--worker-name` | `worker_1` | Worker name reported in logs and telemetry |
| `--no-tls` | `False` | Connects over plaintext TCP without TLS |

---

## 📂 Repository Structure

```
raven_python_miner/
├── distributed/               # Distributed Python Mining System
│   ├── engine_core.py         # KAWPOW math, JIT ProgPOW compiler, Keccak DAG kernels
│   ├── kawpow_server.py       # Stratum pool listener, HTTP DAG server, share dispatcher
│   ├── kawpow_client.py       # Multi-GPU worker, NVML telemetry, VRAM DAG loader
│   └── tls_utils.py           # Native TLS 1.3 context & self-signed certificate generation
├── tests/                     # Test suite & verification scripts
│   ├── test_context.py        # CUDA context initialization check
│   ├── test_nvrtc.py          # NVRTC JIT compilation test
│   ├── test_dag_search_benchmark.py # Batch benchmark
│   └── test_stratum_duplex.py # Stratum protocol duplex communication test
├── docs/                      # Architectural and technical documentation
├── INSTALL.md                 # Complete installation and deployment guide
├── requirements.txt           # Python package dependencies
├── pyproject.toml             # Project build configuration
└── README.md                  # Project overview & documentation
```

---

## 🔒 Security

- Worker-to-server communications support **TLS 1.2 / TLS 1.3**.
- Automatically generates 2048-bit RSA keys and X.509 certificates with Subject Alternative Names (`localhost`, standard IP addresses).
- Custom Certificate Authorities (`--tls-ca`) are supported for enterprise deployments.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This software is intended for educational, testing, and research purposes in cryptocurrency mining and GPU compute architectures. Ensure compliance with your local laws, power regulations, and hosting provider terms of service when running mining workloads.

## ✨ Donate
Generous donations are welcome to RNGyc94iUJnYWCcshAobk2xjUYw7EvLsq4
