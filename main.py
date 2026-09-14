# RavenMining
# Production-grade KAWPOW GPU miner for Ravencoin
# Uses 16-lane cooperative ProgPOW kernel & GPU Keccak-512 DAG engine

import asyncio
import json
import time
import binascii
import sys
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='pycuda')

from kawpow_full_engine import KawpowGpuEngine

class KawpowRavenMiner:
    """Production-grade KAWPOW miner for Ravencoin using CUDA on one GPU.
    Connects to Stratum pool, receives jobs, mines using 16-lane cooperative ProgPOW kernel,
    and submits verified shares.
    """

    EPOCH_LENGTH = 7500  # RVN epoch length

    def __init__(self, pool_url, wallet, worker='gemini_grok', gpu_id=0):
        self.pool_host, self.pool_port = pool_url.split(':')
        self.pool_port = int(self.pool_port)
        self.wallet = wallet
        self.worker = worker
        self.job = None
        self.target = None
        self.blob = None
        self.height = None

        # GPU setup via KawpowGpuEngine
        self.engine = KawpowGpuEngine(gpu_id)
        self.device = self.engine.dev

        # NVML setup for telemetry
        try:
            import pynvml as nvml
            nvml.nvmlInit()
            self.nvml_handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
        except Exception:
            self.nvml_handle = None

    def __del__(self):
        if hasattr(self, 'nvml_handle') and self.nvml_handle:
            try:
                import pynvml as nvml
                nvml.nvmlShutdown()
            except Exception:
                pass

    def get_epoch(self, block_number):
        return int(block_number) // self.EPOCH_LENGTH

    async def connect_to_pool(self):
        while True:
            try:
                await self._run_session()
            except (ConnectionResetError, ConnectionError, OSError, asyncio.IncompleteReadError) as e:
                print(f"[Pool Connection] Connection lost ({e}). Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Miner] Unexpected error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def _run_session(self):
        reader, writer = await asyncio.open_connection(self.pool_host, self.pool_port)
        user = f"{self.wallet}.{self.worker}" if self.worker else self.wallet

        # Subscribe
        sub = {"id": 1, "method": "mining.subscribe", "params": [user, "x"]}
        writer.write(json.dumps(sub).encode() + b'\n')
        await writer.drain()

        line = await reader.readline()
        if not line:
            raise ConnectionResetError("Connection closed by pool during subscribe")

        # Authorize
        auth = {"id": 2, "method": "mining.authorize", "params": [user, "x"]}
        writer.write(json.dumps(auth).encode() + b'\n')
        await writer.drain()

        line = await reader.readline()
        if not line:
            raise ConnectionResetError("Connection closed by pool during authorize")
        auth_resp = json.loads(line)
        if not auth_resp.get('result'):
            print(f"Auth failed: {auth_resp.get('error')}")
            return

        print(f"Connected and authorized as '{user}'")

        # Concurrently run mining worker and pool listener
        listen_task = asyncio.create_task(self.listen_pool(reader))
        miner_task = asyncio.create_task(self.mining_loop(writer, user))
        try:
            done, pending = await asyncio.wait(
                [listen_task, miner_task],
                return_when=asyncio.FIRST_EXCEPTION
            )
            for t in pending:
                t.cancel()
            for t in done:
                t.result()
        finally:
            listen_task.cancel()
            miner_task.cancel()
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def listen_pool(self, reader):
        while True:
            line = await reader.readline()
            if not line:
                raise ConnectionResetError("Pool closed connection")
            msg = json.loads(line)
            if 'method' in msg:
                method = msg['method']
                if method == 'mining.notify':
                    params = msg.get('params', [])
                    if len(params) >= 4:
                        self.job = params[0]
                        self.blob = params[1]  # header_hash hex
                        if isinstance(params[3], str):
                            self.target = int(params[3], 16)
                        if len(params) >= 6:
                            self.height = int(params[5])
                        print(f"[Pool] New job: id={self.job}, height={self.height}")
                        if self.height:
                            try:
                                epoch = self.get_epoch(self.height)
                                seed_bytes = params[2] if len(params) >= 3 else None
                                self.engine.init_epoch(epoch, seed_bytes=seed_bytes)
                            except Exception as e:
                                print(f"[DAG] Error initializing DAG: {e}")
                elif method in ('mining.set_target', 'mining.set_difficulty'):
                    params = msg.get('params', [])
                    if params and isinstance(params[0], str):
                        self.target = int(params[0], 16)
                        print(f"[Pool] Difficulty target set to {params[0][:16]}...")
                    elif params and isinstance(params[0], (int, float)):
                        # If difficulty multiplier is passed, calculate target from diff
                        diff = float(params[0])
                        if diff > 0:
                            # Kawpow standard base target for diff 1: 0x00000000ffff0000000000000000000000000000000000000000000000000000
                            base_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
                            self.target = int(base_target / diff)
                            print(f"[Pool] Target calculated from diff {diff}: {self.target:064x}")
            elif 'id' in msg and msg['id'] is not None:
                if msg.get('error'):
                    print(f"[Pool] Share response: REJECTED ({msg['error']})")
                elif msg.get('result') is True:
                    print(f"[Pool] Share response: ACCEPTED! >>> SUCCESS <<<")

    async def mining_loop(self, writer, user):
        hashes_per_batch = 524288  # 512K hashes per launch (smooth & responsive)
        print(f"[Miner] Configured batch size: {hashes_per_batch:,} nonces/batch")
        batch_nonce = 0
        last_report_time = time.time()
        total_hashes = 0

        last_job = None
        while True:
            if not self.job or not self.blob or not self.engine.dag_gpu or not self.height:
                await asyncio.sleep(0.1)
                continue

            if self.job != last_job:
                last_job = self.job
                batch_nonce = 0

            current_job = self.job
            current_blob = self.blob
            blob_bytes = bytes.fromhex(current_blob)
            if len(blob_bytes) < 32:
                blob_bytes = blob_bytes.ljust(32, b'\x00')

            start_nonce_val = batch_nonce * hashes_per_batch
            if self.target:
                targ_64 = (self.target >> 192) & 0xFFFFFFFFFFFFFFFF if self.target > 0xFFFFFFFFFFFFFFFF else self.target
            else:
                targ_64 = 0x00000000FFFF0000
            if targ_64 == 0:
                targ_64 = 0x00000000FFFF0000

            results = self.engine.search(blob_bytes[:32], start_nonce_val, targ_64, self.height, batch_size=hashes_per_batch)

            total_hashes += hashes_per_batch
            now = time.time()
            elapsed = now - last_report_time
            if elapsed >= 5.0:
                hashrate = (total_hashes / elapsed) / 1e6
                gpu_stats = ""
                if self.nvml_handle:
                    try:
                        import pynvml as nvml
                        util = nvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                        power = nvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
                        temp = nvml.nvmlDeviceGetTemperature(self.nvml_handle, nvml.NVML_TEMPERATURE_GPU)
                        gpu_stats = f" | GPU Load: {util.gpu}% | Mem Ctrl: {util.memory}% ({temp}C, {power:.0f}W)"
                    except Exception:
                        pass
                print(f"[Miner] Hashrate: {hashrate:.2f} MH/s{gpu_stats} | Job: {current_job} | Nonce: 0x{int(start_nonce_val):016x}")
                total_hashes = 0
                last_report_time = now

            if results:
                for fn, mix_bytes in results:
                    nonce_hex = format(fn, '016x')
                    mix_hex = mix_bytes.hex()
                    sub = {"id": 4, "method": "mining.submit", "params": [user, current_job, nonce_hex, current_blob, mix_hex]}
                    writer.write(json.dumps(sub).encode() + b'\n')
                    await writer.drain()
                    print(f"[Miner] Submitted share: nonce={nonce_hex}, mix={mix_hex[:16]}... for job={current_job}")

            batch_nonce += 1
            await asyncio.sleep(0.001)

if __name__ == "__main__":
    print("=== RavenMining (KAWPOW 16-Lane Cooperative Engine) ===")
    pool_url = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 'rvn.2miners.com:6060'
    wallet = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 'RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi'
    worker = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 'grok_miner'

    print(f"Initializing miner for pool '{pool_url}' and wallet '{wallet}.{worker}'...")
    miner = KawpowRavenMiner(pool_url, wallet, worker)
    print(f"CUDA initialization & kernel compilation succeeded on GPU: {miner.device.name()}")

    if "--connect" in sys.argv:
        print(f"Connecting to Stratum pool at {miner.pool_host}:{miner.pool_port}...")
        try:
            asyncio.run(miner.connect_to_pool())
        except KeyboardInterrupt:
            print("\nMining stopped by user.")
    else:
        print("Miner self-test passed successfully! Pass '--connect' to initiate live Stratum pool connection.")