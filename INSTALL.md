# Installation Guide — Raven Python Miner (KAWPOW)

This guide provides step-by-step instructions for installing and running the **Raven Python Miner** on Linux, Windows, and Cloud GPU instances (such as Vast.ai, RunPod, and Lambda Labs).

---

## 1. System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with Compute Capability ≥ 6.0 (Pascal, Turing, Ampere, Ada Lovelace, or newer).
- **VRAM**: Minimum **6 GB VRAM** (Epoch 605+ DAG is ~5.73 GB). 8 GB+ recommended.
- **RAM**: Minimum 8 GB system RAM.
- **Storage**: At least 10 GB free disk space for DAG caching.

### Software Requirements
- **NVIDIA Driver**: Version 525+ (Linux) or 528+ (Windows).
- **CUDA Toolkit**: CUDA 11.8, 12.0, 12.2, 12.4, or 12.8.
- **Python**: Version 3.10, 3.11, or 3.12 (64-bit).
- **C/C++ Compiler**: `gcc`/`g++` (Linux) or MSVC C++ Build Tools (Windows) for PyCUDA JIT compilation.

---

## 2. Cloud GPU Quickstart (Vast.ai / RunPod / Lambda)

Most cloud GPU templates (such as `vastai/base-image_cuda:12.4` or `runpod/pytorch`) already have CUDA and Python pre-configured.

### Quick Setup Commands
Run these commands inside your cloud instance terminal:

```bash
# 1. Update package list and ensure git and build tools are present
apt-get update && apt-get install -y git build-essential python3-dev

# 2. Clone the repository
git clone https://github.com/APS-PPR-2016/raven_python_miner.git
cd raven_python_miner

# 3. Install required Python packages
pip install --upgrade pip
pip install numpy pycuda cuda-python cryptography
```

### Verify GPU Detection
```bash
python3 -c "import pycuda.driver as cuda; cuda.init(); print(f'Detected: {cuda.Device(0).name()} ({cuda.Device(0).total_memory() / (1024**3):.2f} GB VRAM)')"
```

---

## 3. Ubuntu / Debian Linux Setup

### Step 1: Install NVIDIA Driver & CUDA
If you haven't installed CUDA yet:

```bash
# Verify NVIDIA GPU driver is active
nvidia-smi

# Install build dependencies and Python development headers
sudo apt update
sudo apt install -y python3 python3-pip python3-venv build-essential python3-dev
```

### Step 2: Set Up Virtual Environment
```bash
cd raven_python_miner
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Configure CUDA Paths (if needed)
Ensure CUDA compiler (`nvcc`) and libraries are on your `PATH`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
*(Tip: Add these exports to your `~/.bashrc` to make them permanent).*

### Step 4: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 4. Windows 10 / 11 Setup

### Step 1: Install Prerequisites
1. **Python 3.10 – 3.12**: Download from [python.org](https://www.python.org/downloads/). Ensure **"Add python.exe to PATH"** is checked during installation.
2. **NVIDIA CUDA Toolkit**: Download and install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads).
3. **Visual Studio C++ Build Tools**: Download [Visual Studio Community / Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and check **"Desktop development with C++"**.

### Step 2: Set Up Virtual Environment
Open PowerShell:
```powershell
cd raven_python_miner
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

*(Note: If `pip install pycuda` fails on Windows due to missing compiler headers, install pre-compiled PyCUDA wheels or ensure the MSVC `cl.exe` is in your environment PATH).*

---

## 5. Verification & Testing

Verify that your environment can compile CUDA kernels and access the GPU:

```bash
python3 tests/test_context.py
python3 tests/test_nvrtc.py
```

Expected output:
```
[NVRTC] Compilation successful!
[CUDA] Context initialized on GPU: NVIDIA GeForce RTX ...
```

---

## 6. Running the Distributed Miner

### Architecture Overview
1. **Server (`distributed/kawpow_server.py`)**:
   - Connects to Stratum mining pool (e.g., 2Miners, Flypool).
   - Synthesizes and caches the DAG (~5.73 GB for Epoch 605).
   - Serves DAG binary over fast HTTP to workers.
   - Coordinates jobs and aggregates shares over encrypted TLS.

2. **Client (`distributed/kawpow_client.py`)**:
   - Connects to the Server coordinator.
   - Downloads/loads DAG into GPU VRAM.
   - Runs high-throughput JIT-compiled ProgPOW search kernel.
   - Submits valid shares back to the Server.

### Launching on a Single Node (Server + Local GPU Client)

#### Terminal 1 — Coordinator:
```bash
python3 distributed/kawpow_server.py \
  --pool-host rvn.2miners.com \
  --pool-port 6060 \
  --wallet RKAeHM7WDY5w77up4vg94Tf3tC2FjY9G5e \
  --rig-name worker_1 \
  --port 8088 \
  --http-port 8080 \
  --no-tls
```

#### Terminal 2 — GPU Worker:
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

## 7. Troubleshooting

### Q: `pycuda._driver.LogicError: cuMemAlloc failed: out of memory`
- **Cause**: The Ravencoin DAG requires ~5.73 GB of contiguous VRAM. If display outputs or other processes consume VRAM, allocation fails.
- **Fix**: Free VRAM by closing applications, or run on dedicated mining GPUs without display attached (`--gpus 1`).

### Q: `NVRTC JIT compilation failed for period X`
- **Cause**: Missing NVIDIA runtime compilation library or incompatible architecture flag.
- **Fix**: Check that `cuda-python` is installed (`pip install cuda-python`) and that your NVIDIA driver supports the installed CUDA version.

### Q: Server displays `Synthesizing DAG for epoch 605...`
- **Explanation**: This is normal on the first run of a new epoch. The server generates the verified 5.73 GB DAG binary using Keccak-512 and caches it to disk (`.cache_rvn_epoch_605_dag.bin`). Subsequent runs will load from disk in seconds.
