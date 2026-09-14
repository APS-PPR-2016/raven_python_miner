"""
KAWPOW Distributed Mining Client
================================
Connects to kawpow_server.py over Native Python TLS 1.3:
1. Downloads the full DAG from server over HTTP (only if not already cached locally on disk).
2. Loads DAG into GPU VRAM on specified GPU(s) (supports single or multi-GPU mode).
3. Receives live jobs and non-overlapping nonce ranges over an encrypted TLS connection.
4. Mines using high-speed JIT ProgPOW kernels on each local GPU.
5. Submits valid shares back to kawpow_server.py in real-time over the encrypted TLS stream.
6. Seamlessly handles epoch transitions: pauses mining, downloads/loads new DAG, resumes mining.
"""

import asyncio
import json
import os
import sys
import time
import queue
import argparse
import urllib.request
import urllib.error
import threading
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from distributed.engine_core import compile_jit_kernel, get_dataset_num_items
from distributed.tls_utils import get_client_ssl_context
import pycuda.driver as cuda

cuda.init()

class GpuWorker:
    """Manages mining execution on a single GPU with its own isolated CUDA context."""
    def __init__(self, gpu_id, batch_size=524288):
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.dev = cuda.Device(gpu_id)
        self.ctx = self.dev.make_context()
        self.dag_gpu = None
        self.dag_bytes = 0
        self.num_dag_items_2048 = 0
        self.jit_cache = {}
        self.max_found = 1024

        # Pre-allocated GPU search buffers
        self.header_gpu = cuda.mem_alloc(32)
        self.found_nonces_gpu = cuda.mem_alloc(self.max_found * 8)
        self.found_mixes_gpu = cuda.mem_alloc(self.max_found * 32)
        self.found_count_gpu = cuda.mem_alloc(4)

        # Telemetry
        self.total_hashes = 0
        self.last_report = time.time()
        self.hashrate = 0.0

        # Pop context so worker thread can push and own it
        self.ctx.pop()

    def activate_context(self):
        self.ctx.push()

    def deactivate_context(self):
        self.ctx.pop()

    def load_dag_to_vram(self, dag_file_path, num_dag_items_2048):
        self.activate_context()
        try:
            file_size = os.path.getsize(dag_file_path)
            self.dag_bytes = file_size
            self.num_dag_items_2048 = num_dag_items_2048

            if self.dag_gpu is not None:
                print(f"[GPU {self.gpu_id}] Freeing previous epoch DAG buffer ({self.dag_bytes / (1024*1024*1024):.2f} GB)...")
                self.dag_gpu.free()
                self.dag_gpu = None

            print(f"[GPU {self.gpu_id}] Allocating {file_size / (1024*1024*1024):.2f} GB VRAM on {self.dev.name()}...")
            self.dag_gpu = cuda.mem_alloc(file_size)
            t0 = time.time()

            # Stream copy from disk directly to GPU VRAM in 256MB blocks
            chunk_bytes = 256 * 1024 * 1024
            total_copied = 0
            with open(dag_file_path, 'rb') as f:
                while total_copied < file_size:
                    to_read = min(chunk_bytes, file_size - total_copied)
                    chunk = f.read(to_read)
                    cuda.memcpy_htod(int(self.dag_gpu) + total_copied, chunk)
                    total_copied += to_read
                    pct = (total_copied / file_size) * 100
                    print(f"[GPU {self.gpu_id}] Loading DAG to VRAM: {pct:5.1f}% ({total_copied / (1024*1024):.0f} MB)", end="\r")

            print(f"\n[GPU {self.gpu_id}] DAG loaded into VRAM in {time.time() - t0:.2f}s!")
        finally:
            self.deactivate_context()

    def get_search_kernel(self, period):
        if period not in self.jit_cache:
            self.jit_cache[period] = compile_jit_kernel(self.dev, period)
        return self.jit_cache[period]

    def search(self, header_bytes, start_nonce, target_high64, period):
        k_search = self.get_search_kernel(period)
        header_words = np.frombuffer(header_bytes, dtype=np.uint32)
        cuda.memcpy_htod(self.header_gpu, header_words)
        cuda.memset_d32(self.found_count_gpu, 0, 1)

        total_threads = self.batch_size * 16
        threads_per_block = 256
        blocks = (total_threads + threads_per_block - 1) // threads_per_block

        k_search(self.header_gpu, np.uint64(start_nonce), np.uint64(target_high64),
                 self.dag_gpu, np.uint32(self.num_dag_items_2048),
                 np.uint32(self.batch_size),
                 self.found_nonces_gpu, self.found_mixes_gpu, self.found_count_gpu,
                 block=(threads_per_block, 1, 1), grid=(blocks, 1))
        cuda.Context.synchronize()

        cnt = np.zeros(1, dtype=np.uint32)
        cuda.memcpy_dtoh(cnt, self.found_count_gpu)
        n_found = min(int(cnt[0]), self.max_found)

        results = []
        if n_found > 0:
            nonces = np.zeros(n_found, dtype=np.uint64)
            mixes = np.zeros(n_found * 8, dtype=np.uint32)
            cuda.memcpy_dtoh(nonces, self.found_nonces_gpu)
            cuda.memcpy_dtoh(mixes, self.found_mixes_gpu)
            for i in range(n_found):
                mix_bytes = bytes(mixes[i * 8:(i + 1) * 8])
                results.append((int(nonces[i]), mix_bytes))

        self.total_hashes += self.batch_size
        return results

class KawpowClient:
    def __init__(self, server_host, server_port, http_url, gpu_ids, worker_name="worker_1",
                 batch_size=524288, use_tls=True, tls_ca=None):
        self.server_host = server_host
        self.server_port = server_port
        self.http_url = http_url
        self.gpu_ids = gpu_ids
        self.worker_name = worker_name
        self.batch_size = batch_size
        self.use_tls = use_tls
        self.tls_ca = tls_ca

        self.client_id = None
        self.nonce_prefix = 0
        self.active_epoch = None
        self.dag_file = None

        # Active job state
        self.current_job = None
        self.current_blob = None
        self.current_target = 0x00000000FFFF0000
        self.current_height = None
        self.job_lock = threading.Lock()
        self.mining_paused = False

        # Thread-safe outgoing share queue (from GPU search threads -> async TLS network sender)
        self.share_queue = queue.Queue()

        # GPU workers
        self.workers = []
        for gid in self.gpu_ids:
            self.workers.append(GpuWorker(gid, batch_size=self.batch_size))

        self.writer = None
        self.loop = None
        self.running = True
        self.mining_thread_started = False

    def download_dag_file(self, epoch):
        """
        Ensures the DAG file for the given epoch is present on disk.
        If already cached locally, reuses it immediately without downloading.
        Otherwise downloads from server over HTTP, handling generation wait states.
        """
        filename = f".cache_rvn_epoch_{epoch}_dag.bin"
        if os.path.exists(filename) and os.path.getsize(filename) > 1024 * 1024 * 1024:
            print(f"[Client] Found cached DAG file on local disk: {filename} ({os.path.getsize(filename)/(1024*1024*1024):.2f} GB)")
            return filename

        download_url = f"{self.http_url}/dag/download"
        print(f"[Client] Requesting DAG from server: {download_url} -> {filename}...")

        # Retry loop in case server is still synthesizing the DAG
        while self.running:
            try:
                t0 = time.time()
                def report_hook(count, block_size, total_size):
                    downloaded = count * block_size
                    if total_size > 0:
                        pct = min(100.0, (downloaded / total_size) * 100)
                        mb_s = (downloaded / (time.time() - t0 + 0.001)) / (1024 * 1024)
                        print(f"[Client] Downloading DAG: {pct:5.1f}% ({downloaded/(1024*1024):.0f} MB / {total_size/(1024*1024):.0f} MB at {mb_s:.1f} MB/s)", end="\r")

                urllib.request.urlretrieve(download_url, filename, reporthook=report_hook)
                print(f"\n[Client] DAG download complete in {time.time() - t0:.1f}s!")
                return filename

            except urllib.error.HTTPError as e:
                if e.code == 503:
                    print(f"[Client] Server is currently synthesizing DAG for epoch {epoch}. Retrying in 4s...")
                    time.sleep(4)
                else:
                    print(f"[Client] HTTP error downloading DAG: {e}. Retrying in 5s...")
                    time.sleep(5)
            except Exception as e:
                print(f"[Client] Network error downloading DAG: {e}. Retrying in 5s...")
                time.sleep(5)

        return filename

    async def share_sender_task(self):
        """Asynchronous task that reads shares from the thread-safe queue and pushes to server TLS socket."""
        while self.running:
            try:
                share_data = await asyncio.to_thread(self.share_queue.get)
                if share_data is None:
                    break
                if self.writer and not self.writer.is_closing():
                    line = json.dumps(share_data).encode('utf-8') + b'\n'
                    self.writer.write(line)
                    await self.writer.drain()
                    job_id = share_data.get('jid') or share_data.get('job_id')
                    nonce = share_data.get('n') or share_data.get('nonce')
                    print(f"[Client] Sent share to server (TLS): nonce={nonce}, job={job_id}")
                self.share_queue.task_done()
            except Exception as e:
                print(f"[Client] Share sender error: {e}")
                await asyncio.sleep(0.1)

    async def connect_and_run(self):
        self.loop = asyncio.get_running_loop()
        # Start share sender task
        asyncio.create_task(self.share_sender_task())

        while self.running:
            try:
                if self.use_tls:
                    ssl_ctx = get_client_ssl_context(ca_cert=self.tls_ca, allow_self_signed=True)
                    reader, writer = await asyncio.open_connection(self.server_host, self.server_port, ssl=ssl_ctx)
                    print(f"[Client] Connected to KAWPOW server at {self.server_host}:{self.server_port} via TLS 1.3")
                else:
                    reader, writer = await asyncio.open_connection(self.server_host, self.server_port)
                    print(f"[Client] Connected to KAWPOW server at {self.server_host}:{self.server_port} (plaintext)")

                self.writer = writer

                # Register worker name and GPU count
                reg_msg = {
                    'type': 'register',
                    'worker_name': self.worker_name,
                    'gpus': len(self.gpu_ids)
                }
                writer.write(json.dumps(reg_msg).encode() + b'\n')
                await writer.drain()

                # Start mining thread once
                if not self.mining_thread_started:
                    self.mining_thread_started = True
                    mining_thread = threading.Thread(target=self.mining_loop_thread, daemon=True)
                    mining_thread.start()

                # Listen for server messages (jobs, epoch updates)
                while self.running:
                    line = await reader.readline()
                    if not line:
                        raise ConnectionResetError("Server closed connection")
                    msg = json.loads(line.decode('utf-8'))
                    mtype = msg.get('type')

                    if mtype == 'init':
                        self.client_id = msg.get('c_id') or msg.get('client_id')
                        self.nonce_prefix = msg.get('np') or msg.get('nonce_prefix', 0)
                        epoch = msg.get('ep') or msg.get('epoch')
                        print(f"[Client] Initialized as Client {self.client_id} (nonce prefix: 0x{self.nonce_prefix:016x})")
                        if epoch is not None and epoch != self.active_epoch:
                            self.prepare_epoch(epoch)

                    elif mtype == 'epoch_transition':
                        epoch = msg.get('ep') or msg.get('epoch')
                        status = msg.get('status', 'generating')
                        print(f"[Client] *** Server reported epoch transition to {epoch} (status: {status}) ***")
                        self.mining_paused = True
                        if status == 'ready':
                            self.prepare_epoch(epoch)

                    elif mtype == 'epoch_ready':
                        epoch = msg.get('ep') or msg.get('epoch')
                        print(f"[Client] Server announced epoch {epoch} DAG is ready!")
                        self.prepare_epoch(epoch)

                    elif mtype == 'job':
                        epoch = msg.get('ep') or msg.get('epoch')
                        if epoch is not None and epoch != self.active_epoch:
                            print(f"[Client] Received job with new epoch {epoch}. Switching epoch...")
                            self.prepare_epoch(epoch)

                        with self.job_lock:
                            self.current_job = msg.get('jid') or msg.get('job_id')
                            self.current_blob = msg.get('bl') or msg.get('blob')
                            targ_val = msg.get('t') or msg.get('target')
                            if isinstance(targ_val, str):
                                targ_val = int(targ_val, 16)
                            if targ_val > 0xFFFFFFFFFFFFFFFF:
                                self.current_target = (targ_val >> 192) & 0xFFFFFFFFFFFFFFFF
                            else:
                                self.current_target = targ_val
                            if self.current_target == 0:
                                self.current_target = 0x00000000FFFF0000
                            self.current_height = msg.get('h') or msg.get('height')

            except (ConnectionResetError, ConnectionError, OSError) as e:
                print(f"[Client] Server connection lost ({e}). Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"[Client] Unexpected error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    def prepare_epoch(self, epoch):
        """Pauses mining, downloads or reuses DAG, reallocates VRAM, and resumes mining."""
        self.mining_paused = True
        print(f"[Client] Preparing epoch {epoch}...")
        self.active_epoch = epoch

        dag_path = self.download_dag_file(epoch)
        self.dag_file = dag_path

        full_items = get_dataset_num_items(epoch)
        num_dag_items_2048 = full_items // 2

        for w in self.workers:
            w.load_dag_to_vram(dag_path, num_dag_items_2048)

        print(f"[Client] Epoch {epoch} ready! Resuming mining.")
        self.mining_paused = False

    def enqueue_share(self, job_id, nonce_hex, blob, mix_hex):
        """Thread-safe enqueueing of found shares from GPU workers."""
        share_msg = {
            'type': 'share',
            'jid': job_id,
            'job_id': job_id,
            'n': nonce_hex,
            'nonce': nonce_hex,
            'h': blob,
            'header': blob,
            'm': mix_hex,
            'mix_hash': mix_hex
        }
        self.share_queue.put(share_msg)

    def mining_loop_thread(self):
        """Runs the high-speed CUDA mining loop across all designated GPUs."""
        batch_counter = 0
        last_job = None
        last_report = time.time()

        while self.running:
            if self.mining_paused or not self.workers[0].dag_gpu:
                time.sleep(0.05)
                continue

            with self.job_lock:
                c_job = self.current_job
                c_blob = self.current_blob
                c_height = self.current_height
                c_target = self.current_target

            if not c_job or not c_blob or not c_height:
                time.sleep(0.05)
                continue

            if c_job != last_job:
                last_job = c_job
                batch_counter = 0

            blob_bytes = bytes.fromhex(c_blob).ljust(32, b'\x00')[:32]
            period = c_height // 3

            # Search across local GPU workers
            for g_idx, w in enumerate(self.workers):
                w.activate_context()
                try:
                    base_gpu_nonce = self.nonce_prefix + (g_idx << 36)
                    start_nonce = base_gpu_nonce + (batch_counter * self.batch_size)

                    results = w.search(blob_bytes, start_nonce, c_target, period)
                    if results:
                        for fn, mix_bytes in results:
                            nonce_hex = format(fn, '016x')
                            mix_hex = mix_bytes.hex()
                            print(f"[Client] >>> FOUND SHARE on GPU {w.gpu_id}! Nonce: {nonce_hex}, Mix: {mix_hex[:16]}... <<<")
                            self.enqueue_share(c_job, nonce_hex, c_blob, mix_hex)
                finally:
                    w.deactivate_context()

            batch_counter += 1

            now = time.time()
            if now - last_report >= 5.0:
                dt = now - last_report
                total_h = sum(w.total_hashes for w in self.workers)
                agg_mhs = (total_h / dt) / 1e6
                for w in self.workers:
                    w.total_hashes = 0
                gpu_str = ", ".join(f"GPU {w.gpu_id}" for w in self.workers)
                print(f"[Client] Total Hashrate: {agg_mhs:.2f} MH/s ({gpu_str}) | Job: {c_job} | Batch: #{batch_counter:,}")
                last_report = now

            time.sleep(0.001)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAWPOW Distributed Mining Client")
    parser.add_argument("--server", default="127.0.0.1:8088", help="Mining server host:port")
    parser.add_argument("--http-server", default="http://127.0.0.1:8080", help="HTTP DAG download server URL")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU device IDs (e.g. 0 or 0,1)")
    parser.add_argument("--worker", default="worker_node1", help="Worker name")
    parser.add_argument("--batch-size", type=int, default=524288, help="Nonces per batch per GPU")
    parser.add_argument("--no-tls", action="store_true", help="Disable TLS and connect in plaintext")
    parser.add_argument("--tls-ca", default=None, help="Optional CA certificate to verify server")
    args = parser.parse_args()

    s_host, s_port = args.server.split(':')
    gpu_list = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]

    print(f"=== KAWPOW Mining Client ({len(gpu_list)} GPU(s): {gpu_list}) ===")
    client = KawpowClient(
        server_host=s_host,
        server_port=int(s_port),
        http_url=args.http_server,
        gpu_ids=gpu_list,
        worker_name=args.worker,
        batch_size=args.batch_size,
        use_tls=not args.no_tls,
        tls_ca=args.tls_ca
    )

    try:
        asyncio.run(client.connect_and_run())
    except KeyboardInterrupt:
        client.running = False
        print("\nClient stopped by user.")
