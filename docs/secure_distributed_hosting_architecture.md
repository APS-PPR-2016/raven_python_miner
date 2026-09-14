# Secure Distributed Architecture & Hosting Report
**Project:** KAWPOW Distributed Mining Network  
**Topic:** Hardening and Securing Server–Client Communication across External IPs  
**Document Location:** `docs/secure_distributed_hosting_architecture.md`  
**Date:** September 2026  

---

## 1. Executive Summary & Problem Analysis

In the initial working prototype of the distributed KAWPOW miner:
1. **HTTP Server (Port 8080):** Serves the ~5.7–6.2 GB DAG dataset over unencrypted, unauthenticated plain HTTP.
2. **Mining Coordinator (Port 8088):** Streams active jobs, block headers, and submits shares using plain JSON-Lines over raw TCP.

When transitioning from local testing (`127.0.0.1` / private LAN) to hosting `kawpow_server.py` on an **external public IP** (cloud VPS, dedicated colocation, or a remote server) with clients distributed across different geographical locations, several serious vulnerabilities arise:

### Threat Vectors on a Public IP
* **Man-in-the-Middle (MITM) & Share Hijacking:** An attacker in the transit network (ISP, open Wi-Fi, malicious router) can intercept mining shares, modify target addresses, or alter job headers to direct GPU hashrate to their own wallet.
* **Network Eavesdropping:** Plaintext traffic exposes pool credentials, wallet addresses, worker names, active network topologies, and mining performance telemetry.
* **Bandwidth Exhaustion & DDoS:** A naked HTTP endpoint serving 6 GB files publicly can easily be abused by scrapers, crawlers, or bad actors to saturate the server's egress bandwidth, incurring heavy data costs or causing Denial of Service.
* **Rogue Client Ingestion:** Without an authentication layer, unauthorized machines could connect to the server, consume partition nonces without mining, or flood the server with malformed shares causing pool ban.
* **DAG File Tampering / Corruption:** If an adversary modifies or corrupts the binary DAG stream, client GPUs will mine on invalid DAG data, resulting in 100% rejected shares.

---

## 2. Architecture Comparison for External Hosting

There are three primary architectural patterns to secure this infrastructure across external IPs:

```
+-----------------------------------------------------------------------------------+
|                            ARCHITECTURE TAXONOMY                                  |
+-----------------------------------+-----------------------------------------------+
| Approach                          | Description & Suitability                     |
+-----------------------------------+-----------------------------------------------+
| Option 1: WireGuard / Tailscale   | Encrypted overlay mesh network.               |
| (Recommended for Private Mining)  | Highest security, zero open public ports.     |
+-----------------------------------+-----------------------------------------------+
| Option 2: Reverse Proxy + TLS     | Nginx/Caddy terminating HTTPS & WSS/TLS-TCP.  |
| (Industry Standard for Services)  | Ideal if managing public or multi-tenant rigs.|
+-----------------------------------+-----------------------------------------------+
| Option 3: Native Python TLS & mTLS| Python `ssl` module inside server/client.     |
| (Zero External Dependencies)      | Built directly into the Python binaries.      |
+-----------------------------------+-----------------------------------------------+
```

---

## 3. Detailed Architectural Approaches

### Option 1: Overlay Mesh VPN (WireGuard / Tailscale) — *Recommended for Dedicated Rigs*

If all client machines are owned or managed by you, an encrypted overlay network is the cleanest, most performant, and most secure solution.

#### How It Works:
* The server and clients join a private WireGuard or Tailscale overlay mesh (e.g. `100.64.0.1` for server, `100.64.0.2`, `100.64.0.3` for clients).
* The server binds `kawpow_server.py` **only** to its internal VPN interface IP (`100.64.0.1`), never to `0.0.0.0`.
* The public IP firewall drops all traffic on ports 8080 and 8088 from the internet.

```
       [ Public Internet ]
               |
  (All Public Ports Blocked)
               |
  +--------------------------+
  | Server (WireGuard Node)  | <=== ChaCha20-Poly1305 Encrypted Tunnel ===> [ Remote Client Node ]
  | IP: 100.64.0.1           |                                              | IP: 100.64.0.2
  | HTTP: 100.64.0.1:8080    |                                              | Runs kawpow_client.py
  | TCP:  100.64.0.1:8088    |                                              |
  +--------------------------+
```

#### Advantages:
1. **Zero Attack Surface:** No public ports are exposed on the external IP. Port scanners see closed ports.
2. **Kernel-Level WireGuard Performance:** Negligible CPU impact, line-rate throughput for 6 GB DAG downloads.
3. **Automatic NAT Traversal:** Clients behind home routers or strict firewalls connect effortlessly.
4. **Built-in Mutual Authentication:** Keys are exchanged beforehand; rogue machines cannot even establish a handshake.

---

### Option 2: Reverse Proxy with TLS (Nginx / Caddy) + Token Auth

If you must host on a public domain/IP without installing VPN clients on every machine (e.g., standard rigs, diverse OS environments):

```
                     [ Public Internet ]
                              |
                     HTTPS:443 / WSS:8443
                              |
                      +---------------+
                      | Nginx / Caddy | <--- Let's Encrypt TLS 1.3
                      +-------+-------+
                              | Local Loopback (127.0.0.1)
          +-------------------+-------------------+
          |                                       |
+---------v----------+                 +----------v---------+
| DAG HTTP Server    |                 | Coordinator TCP    |
| Port 8080          |                 | Port 8088          |
+--------------------+                 +--------------------+
```

#### Key Implementation Pillars:
1. **HTTPS (Port 443) with Kernel `sendfile`:**
   * Nginx directly hosts the `.cache_rvn_epoch_*_dag.bin` directory using native Linux kernel `sendfile` and `tcp_nopush`. This saturates 1 Gbps / 10 Gbps uplinks with ~0% CPU utilization.
   * Enables HTTP Range requests (`Accept-Ranges: bytes`), allowing clients to resume interrupted 6 GB downloads rather than starting over.
2. **Stream / WebSocket Proxying for Port 8088:**
   * Nginx `stream` module wraps the TCP socket in TLS 1.3:
     ```nginx
     stream {
         server {
             listen 8443 ssl;
             proxy_pass 127.0.0.1:8088;
             ssl_certificate /etc/letsencrypt/live/miner.domain.com/fullchain.pem;
             ssl_certificate_key /etc/letsencrypt/live/miner.domain.com/privkey.pem;
             ssl_protocols TLSv1.3;
         }
     }
     ```
3. **Bearer Token & Rate Limiting:**
   * Require an authorization header (`Authorization: Bearer <secret_token>`) for DAG downloads.
   * Rate-limit download requests to prevent DDoS (`limit_req_zone`).

---

### Option 3: Built-In Python TLS Encryption (`ssl` standard library)

For a standalone setup with no external proxies (Nginx) or VPNs:

```
[ Client Machine ]                                         [ Server Machine (Public IP) ]
kawpow_client.py                                           kawpow_server.py
ssl_context = ssl.create_default_context()                 ssl_context = ssl.SSLContext(PROTOCOL_TLS_SERVER)
ssl_context.load_verify_locations('ca.crt')                ssl_context.load_cert_chain('server.crt', 'server.key')
         |                                                          |
         +=========== Encrypted TLS 1.3 TCP Socket =================+
         | (Jobs, Nonces, Shares transmitted over encrypted stream) |
```

#### Protocol Hardening:
* Wrap `asyncio.start_server(..., ssl=server_ssl_ctx)`.
* Wrap client `asyncio.open_connection(..., ssl=client_ssl_ctx)`.
* **Mutual TLS (mTLS):** The server requires the client to present a valid client certificate signed by your private CA. Unauthorized clients are rejected during the TLS handshake before Python parses any data.

---

## 4. End-to-End Security Checklist

| Category | Vulnerability | Recommended Solution |
| :--- | :--- | :--- |
| **Transport Encryption** | Eavesdropping / ISP sniffing | Enforce TLS 1.3 (ChaCha20-Poly1305 / AES-256-GCM) across both HTTP and TCP streams. |
| **Authentication** | Unauthorized rigs / spoofed workers | Pre-shared HMAC-SHA256 Token or mTLS Client Certificates. |
| **Data Integrity** | Corrupted or poisoned DAG binaries | Compute BLAKE3 / SHA-256 checksum during DAG export on server; client verifies checksum prior to GPU VRAM upload. |
| **Bandwidth Abuse** | Public scrapers downloading 6 GB DAG | Restrict download endpoint via firewall, bearer tokens, or pre-signed temporary URLs. |
| **Replay Attacks** | Resending stale shares | Include timestamp + monotonic sequence number in share payloads. |
| **OS / Network Perimeter** | Port probing and brute-forcing | Configure `ufw` / `iptables` or cloud security groups to restrict ingress to known worker IPs or VPN subnets. |

---

## 5. DAG Checksum & Validation Architecture

To eliminate the risk of mining on a corrupted or tampered 6 GB DAG file over an external connection:

1. **Server-Side Hash Publication:**
   When the server finishes generating `.cache_rvn_epoch_{epoch}_dag.bin`, it computes a fast BLAKE3 or SHA-256 hash and publishes it at `/dag/info`:
   ```json
   {
     "epoch": 605,
     "dag_bytes": 6148849024,
     "sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
   }
   ```
2. **Client-Side Validation:**
   The client downloads the file, computes the hash locally, and confirms equality before calling `cuda.memcpy_htod()` to load it into VRAM. If the hash fails, the download is retried automatically.

---

## 6. Recommended Action Plan

For deploying the current codebase to an external IP:

1. **Tier 1 (Fastest, High Security — 15 mins): Tailscale / WireGuard**
   * Install Tailscale on the server and client machines.
   * Bind `kawpow_server.py` to the Tailscale interface IP.
   * Connect clients via the private Tailscale IP. Zero open public ports required.
2. **Tier 2 (Production Scale / Multi-User): Nginx Reverse Proxy**
   * Point a domain (e.g., `mining.yourdomain.com`) to the server.
   * Issue a Let's Encrypt TLS certificate via Certbot.
   * Use Nginx to serve the 6 GB DAG over HTTPS and proxy the TCP coordinator over TLS.
   * Add a shared secret token in `kawpow_client.py` and `kawpow_server.py`.
3. **Tier 3 (Protocol Hardening): DAG Integrity Verification**
   * Integrate SHA-256 / BLAKE3 checksum verification into the client's `download_dag_file` routine.

---
*Report generated for the KAWPOW Mining Infrastructure.*
