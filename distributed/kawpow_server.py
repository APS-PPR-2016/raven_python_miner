"""
KAWPOW Mining Server (Coordinator & DAG Distributor)
===================================================
1. Connects to the upstream Stratum pool (e.g. rvn.2miners.com:6060).
2. Generates Keccak-512 light cache and full DAG (cached to disk as .cache_rvn_epoch_{epoch}_dag.bin).
3. Serves the full DAG over high-throughput HTTP for network clients to download/cache.
4. Manages connected mining clients (kawpow_client.py) over an async TCP protocol:
   - Partitions non-overlapping 64-bit nonce ranges per worker.
   - Broadcasts new pool jobs with sub-millisecond latency.
   - Collects client-submitted shares into an instant queue and relays them to the Stratum pool.
"""

import asyncio
import json
import os
import sys
import time
import argparse
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from distributed.engine_core import (
    get_cache_num_items, get_dataset_num_items, compile_dag_engine
)
import pycuda.driver as cuda
import pycuda.autoinit

class KawpowMiningServer:
    def __init__(self, pool_host, pool_port, wallet, worker, http_port=8080, client_port=8088, device_id=0):
        self.pool_host = pool_host
        self.pool_port = pool_port
        self.wallet = wallet
        self.worker = worker
        self.http_port = http_port
        self.client_port = client_port
        self.device_id = device_id

        self.current_epoch = None
        self.dag_file = None
        self.dag_bytes = 0

        # Pool state
        self.pool_writer = None
        self.current_job = None
        self.current_blob = None
        self.current_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
        self.current_height = None
        self.current_epoch_len = 7500

        # Client management
        self.clients = {}  # client_id -> {'writer': w, 'worker_name': name, 'gpus': count, 'shares': 0}
        self.next_client_id = 1

        # Share queue for instant pool forwarding
        self.share_queue = asyncio.Queue()
        self.stats = {
            'shares_valid': 0,
            'shares_invalid': 0,
            'total_shares_submitted': 0
        }

    def get_epoch(self, block_number):
        return int(block_number) // self.current_epoch_len

    def ensure_dag_on_disk(self, epoch, seed_bytes=None):
        """Generates the DAG on GPU if not already on disk, and saves it for network distribution."""
        if self.current_epoch == epoch and self.dag_file and os.path.exists(self.dag_file):
            return

        self.current_epoch = epoch
        num_cache_items = get_cache_num_items(epoch)
        full_items = get_dataset_num_items(epoch)
        total_half_items = full_items * 2
        dag_bytes = total_half_items * 64
        self.dag_bytes = dag_bytes
        self.dag_file = f".cache_rvn_epoch_{epoch}_dag.bin"

        cache_bytes = num_cache_items * 64
        cache_file = f".cache_rvn_epoch_{epoch}_keccak.bin"

        print(f"[Server] Epoch {epoch}: DAG size is {dag_bytes / (1024*1024*1024):.2f} GB ({dag_bytes:,} bytes)")

        if os.path.exists(self.dag_file) and os.path.getsize(self.dag_file) == dag_bytes:
            print(f"[Server] Found verified DAG file on disk: {self.dag_file}")
            return

        print(f"[Server] DAG file not found on disk. Synthesizing on GPU {self.device_id}...")
        dev = cuda.Device(self.device_id)
        dag_mod = compile_dag_engine(dev)
        k_cache = dag_mod.get_function("build_light_cache_gpu")
        k_dag = dag_mod.get_function("generate_dag_kernel_keccak")

        # 1. Light cache
        cache_gpu = cuda.mem_alloc(cache_bytes)
        if os.path.exists(cache_file) and os.path.getsize(cache_file) == cache_bytes:
            print(f"[Server] Loading light cache from {cache_file}...")
            cache_host = np.fromfile(cache_file, dtype=np.uint8)
            cuda.memcpy_htod(cache_gpu, cache_host)
        else:
            if seed_bytes is None:
                seed = np.zeros(4, dtype=np.uint64)
            elif isinstance(seed_bytes, str):
                seed = np.frombuffer(bytes.fromhex(seed_bytes), dtype=np.uint64)
            else:
                seed = np.frombuffer(seed_bytes, dtype=np.uint64)

            seed_gpu = cuda.mem_alloc(32)
            cuda.memcpy_htod(seed_gpu, seed)
            print(f"[Server] Generating {num_cache_items:,} Keccak light cache items...")
            k_cache(seed_gpu, cache_gpu, np.uint32(num_cache_items), block=(1, 1, 1), grid=(1, 1))
            cuda.Context.synchronize()
            seed_gpu.free()

            cache_host = np.zeros(cache_bytes, dtype=np.uint8)
            cuda.memcpy_dtoh(cache_host, cache_gpu)
            cache_host.tofile(cache_file)
            print(f"[Server] Saved light cache to {cache_file}")

        # 2. Synthesize DAG in chunks and stream directly to disk file
        dag_gpu = cuda.mem_alloc(dag_bytes)
        threads = 256
        chunk_size = 4000000  # ~256 MB per chunk
        total_chunks = (total_half_items + chunk_size - 1) // chunk_size
        t0 = time.time()

        print(f"[Server] Synthesizing {dag_bytes / (1024*1024*1024):.2f} GB DAG in {total_chunks} chunks...")
        for chunk_idx, start_item in enumerate(range(0, total_half_items, chunk_size)):
            count = min(chunk_size, total_half_items - start_item)
            blocks = (count + threads - 1) // threads
            k_dag(cache_gpu, np.uint32(num_cache_items), dag_gpu,
                  np.uint32(start_item), np.uint32(count),
                  block=(threads, 1, 1), grid=(blocks, 1))
            cuda.Context.synchronize()
            pct = ((start_item + count) / total_half_items) * 100
            print(f"[Server] DAG generation: {pct:5.1f}% complete ({chunk_idx + 1}/{total_chunks})", end="\r")

        print(f"\n[Server] DAG synthesis complete in {time.time() - t0:.2f}s! Exporting to disk: {self.dag_file}...")
        cache_gpu.free()

        # Stream copy from GPU to disk in 256MB blocks
        with open(self.dag_file, 'wb') as f:
            chunk_bytes = 256 * 1024 * 1024
            buf = np.zeros(chunk_bytes // 4, dtype=np.uint32)
            total_copied = 0
            while total_copied < dag_bytes:
                to_copy = min(chunk_bytes, dag_bytes - total_copied)
                if to_copy < chunk_bytes:
                    buf = np.zeros(to_copy // 4, dtype=np.uint32)
                cuda.memcpy_dtoh(buf, int(dag_gpu) + total_copied)
                f.write(buf.tobytes())
                total_copied += to_copy
                print(f"[Server] Exported {total_copied / (1024*1024):.0f} MB / {dag_bytes / (1024*1024):.0f} MB to disk...", end="\r")

        print(f"\n[Server] Successfully saved verified DAG file to {self.dag_file} ({os.path.getsize(self.dag_file):,} bytes)")
        dag_gpu.free()

    # --- HTTP File Server for DAG Distribution ---
    async def handle_http_request(self, reader, writer):
        try:
            req_line = await reader.readline()
            if not req_line:
                writer.close()
                return
            req_str = req_line.decode('utf-8', errors='ignore')
            parts = req_str.split()
            if len(parts) < 2:
                writer.close()
                return
            method, path = parts[0], parts[1]

            # Read remaining headers
            while True:
                h_line = await reader.readline()
                if not h_line or h_line == b'\r\n' or h_line == b'\n':
                    break

            if path == '/dag/info':
                resp_data = json.dumps({
                    'epoch': self.current_epoch,
                    'dag_bytes': self.dag_bytes,
                    'filename': self.dag_file
                }).encode('utf-8')
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(resp_data)).encode() + b"\r\n\r\n" + resp_data)
                await writer.drain()
            elif path.startswith('/dag/download'):
                if not self.dag_file or not os.path.exists(self.dag_file):
                    not_found = b"HTTP/1.1 404 Not Found\r\n\r\nDAG not ready yet"
                    writer.write(not_found)
                    await writer.drain()
                else:
                    file_size = os.path.getsize(self.dag_file)
                    headers = (
                        f"HTTP/1.1 200 OK\r\n"
                        f"Content-Type: application/octet-stream\r\n"
                        f"Content-Length: {file_size}\r\n"
                        f"Content-Disposition: attachment; filename={os.path.basename(self.dag_file)}\r\n"
                        f"Connection: close\r\n\r\n"
                    ).encode('utf-8')
                    writer.write(headers)
                    await writer.drain()

                    # Stream file in 8MB chunks
                    chunk_size = 8 * 1024 * 1024
                    with open(self.dag_file, 'rb') as f:
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            writer.write(chunk)
                            await writer.drain()
            else:
                writer.write(b"HTTP/1.1 404 Not Found\r\n\r\nInvalid endpoint")
                await writer.drain()
        except Exception as e:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # --- Client TCP Coordinator ---
    async def handle_client(self, reader, writer):
        client_id = self.next_client_id
        self.next_client_id += 1
        addr = writer.get_extra_info('peername')
        client_info = {'writer': writer, 'worker_name': f'worker_{client_id}', 'gpus': 1, 'shares': 0, 'id': client_id}
        self.clients[client_id] = client_info
        print(f"[Server] Client {client_id} connected from {addr}")

        try:
            # Send initial state and nonce partition
            # Each client is assigned a 40-bit partition prefix (client_id << 40)
            nonce_prefix = client_id << 40
            init_msg = {
                'type': 'init',
                'c_id': client_id,
                'np': nonce_prefix,
                'ep': self.current_epoch,
                'dh': f"http://localhost:{self.http_port}/dag/download"
            }
            writer.write(json.dumps(init_msg).encode() + b'\n')
            await writer.drain()

            # If there's an active job, send it immediately
            if self.current_job:
                job_msg = {
                    'type': 'job',
                    'jid': self.current_job,
                    'bl': self.current_blob,
                    't': self.current_target,
                    'h': self.current_height,
                    'ep': self.current_epoch
                }
                writer.write(json.dumps(job_msg).encode() + b'\n')
                await writer.drain()

            while True:
                line = await reader.readline()
                if not line:
                    break
                msg = json.loads(line.decode('utf-8'))
                mtype = msg.get('type')

                if mtype == 'register':
                    client_info['worker_name'] = msg.get('worker_name', client_info['worker_name'])
                    client_info['gpus'] = msg.get('gpus', 1)
                    print(f"[Server] Client {client_id} registered as '{client_info['worker_name']}' ({client_info['gpus']} GPU(s))")

                elif mtype == 'share':
                    # Client found a valid share -> put on queue for instant pool submission
                    share_data = {
                        'client_id': client_id,
                        'worker_name': client_info['worker_name'],
                        'job_id': msg['jid'],
                        'nonce': msg['n'],
                        'header': msg['h'],
                        'mix_hash': msg['m']
                    }
                    await self.share_queue.put(share_data)

                elif mtype == 'hashrate':
                    hr = msg.get('hr', 0.0)
                    client_info['hashrate'] = hr

        except (ConnectionResetError, ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception as e:
            print(f"[Server] Client {client_id} error: {e}")
        finally:
            print(f"[Server] Client {client_id} disconnected.")
            if client_id in self.clients:
                del self.clients[client_id]
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # --- Stratum Pool Dispatcher & Share Queue Consumer ---
    async def process_share_queue(self):
        """Asynchronously pulls shares from the queue and sends them to the pool immediately."""
        while True:
            share = await self.share_queue.get()
            try:
                if self.pool_writer and not self.pool_writer.is_closing():
                    user = f"{self.wallet}.{share['worker_name']}"
                    sub_msg = {
                        "id": 4,
                        "method": "mining.submit",
                        "params": [user, share['job_id'], share['nonce'], share['header'], share['mix_hash']]
                    }
                    self.pool_writer.write(json.dumps(sub_msg).encode() + b'\n')
                    await self.pool_writer.drain()
                    self.stats['total_shares_submitted'] += 1
                    print(f"[Server] >>> Submitted share from {share['worker_name']}: nonce={share['nonce']}, mix={share['mix_hash'][:16]}... for job={share['job_id']}")
                else:
                    print("[Server] Warning: Pool writer not connected; share queued could not be delivered.")
            except Exception as e:
                print(f"[Server] Error submitting share to pool: {e}")
            finally:
                self.share_queue.task_done()

    async def broadcast_job(self):
        """Broadcasts active pool job to all connected clients."""
        if not self.clients or not self.current_job:
            return
        job_msg = {
            'type': 'job',
            'job_id': self.current_job,
            'blob': self.current_blob,
            'target': self.current_target,
            'height': self.current_height,
            'epoch': self.current_epoch
        }
        msg_bytes = json.dumps(job_msg).encode() + b'\n'
        for cid, c in list(self.clients.items()):
            try:
                c['writer'].write(msg_bytes)
                await c['writer'].drain()
            except Exception:
                pass

    # --- Stratum Pool Connection Session ---
    async def connect_to_pool(self):
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.pool_host, self.pool_port)
                self.pool_writer = writer
                user = f"{self.wallet}.{self.worker}"

                # Subscribe
                writer.write(json.dumps({"id": 1, "method": "mining.subscribe", "params": [user, "x"]}).encode() + b'\n')
                await writer.drain()
                sub_line = await reader.readline()
                if not sub_line: raise ConnectionResetError("Pool closed connection")

                # Authorize
                writer.write(json.dumps({"id": 2, "method": "mining.authorize", "params": [user, "x"]}).encode() + b'\n')
                await writer.drain()
                auth_line = await reader.readline()
                if not auth_line: raise ConnectionResetError("Pool closed connection")
                auth_resp = json.loads(auth_line)
                if not auth_resp.get('result'):
                    print(f"[Server] Pool auth failed: {auth_resp.get('error')}")
                    return

                print(f"[Server] Connected & authorized to Stratum pool at {self.pool_host}:{self.pool_port}")

                # Listen for pool jobs
                while True:
                    line = await reader.readline()
                    if not line:
                        raise ConnectionResetError("Pool connection closed")
                    msg = json.loads(line)

                    if 'method' in msg:
                        method = msg['method']
                        if method == 'mining.notify':
                            params = msg.get('params', [])
                            if len(params) >= 4:
                                self.current_job = params[0]
                                self.current_blob = params[1]
                                if isinstance(params[3], str):
                                    self.current_target = int(params[3], 16)
                                if len(params) >= 6:
                                    self.current_height = int(params[5])

                                epoch = self.get_epoch(self.current_height)
                                seed_bytes = params[2] if len(params) >= 3 else None
                                print(f"[Pool] Job: id={self.current_job}, height={self.current_height}, epoch={epoch}")

                                # Ensure DAG file is generated on disk
                                self.ensure_dag_on_disk(epoch, seed_bytes)

                                # Broadcast job to all connected clients
                                await self.broadcast_job()

                        elif method in ('mining.set_target', 'mining.set_difficulty'):
                            params = msg.get('params', [])
                            if params and isinstance(params[0], str):
                                self.current_target = int(params[0], 16)

                    elif 'id' in msg and msg['id'] is not None:
                        if msg.get('error'):
                            self.stats['shares_invalid'] += 1
                            print(f"[Pool] Share REJECTED: {msg['error']}")
                        elif msg.get('result') is True:
                            self.stats['shares_valid'] += 1
                            print(f"[Pool] >>> Share ACCEPTED! <<< (Valid: {self.stats['shares_valid']}, Invalid: {self.stats['shares_invalid']})")

            except (ConnectionResetError, ConnectionError, OSError, asyncio.IncompleteReadError) as e:
                print(f"[Server] Pool connection lost ({e}). Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"[Server] Unexpected pool error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def start(self):
        # 1. Start HTTP DAG file server
        http_server = await asyncio.start_server(self.handle_http_request, '0.0.0.0', self.http_port)
        print(f"[Server] HTTP DAG Server listening on http://0.0.0.0:{self.http_port}")

        # 2. Start Client Coordinator TCP server
        client_server = await asyncio.start_server(self.handle_client, '0.0.0.0', self.client_port)
        print(f"[Server] Mining Client Coordinator listening on port {self.client_port}")

        # 3. Start share submission consumer queue
        asyncio.create_task(self.process_share_queue())

        # 4. Connect to upstream Stratum pool
        await self.connect_to_pool()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAWPOW Mining Server & DAG Distributor")
    parser.add_argument("--pool", default="rvn.2miners.com:6060", help="Stratum pool host:port")
    parser.add_argument("--wallet", default="RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi", help="Ravencoin wallet address")
    parser.add_argument("--worker", default="kawpow_server", help="Worker name")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP port for DAG downloads")
    parser.add_argument("--client-port", type=int, default=8088, help="TCP port for mining clients")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID for DAG generation")
    args = parser.parse_args()

    host, port = args.pool.split(':')
    server = KawpowMiningServer(host, int(port), args.wallet, args.worker, args.http_port, args.client_port, args.gpu)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
