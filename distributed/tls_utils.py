"""
TLS Utilities for KAWPOW Distributed Mining
===========================================
Provides native Python TLS context configuration and automatic self-signed
certificate generation using the cryptography library.
Ensures zero-config, encrypted communication between server and clients over external IPs.
"""

import os
import ssl
import datetime
import ipaddress
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

def ensure_tls_credentials(cert_path="server.crt", key_path="server.key", host="localhost"):
    """
    Ensures that a TLS certificate and private key exist.
    If missing, automatically generates a self-signed X.509 certificate
    valid for 10 years with Subject Alternative Names (SAN) for localhost,
    all IP addresses, and common hostnames.
    """
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    print(f"[TLS] Generating self-signed TLS credentials: {cert_path}, {key_path}...")
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Global"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Network"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "KAWPOW Distributed Miner"),
        x509.NameAttribute(NameOID.COMMON_NAME, host or "kawpow-server"),
    ])

    san_list = [
        x509.DNSName("localhost"),
        x509.DNSName("kawpow-server"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv4Address("0.0.0.0")),
    ]
    if host and host not in ("localhost", "127.0.0.1", "0.0.0.0"):
        try:
            san_list.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            san_list.append(x509.DNSName(host))

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .sign(key, hashes.SHA256())
    )

    # Write private key
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # Write certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"[TLS] Certificate and key generated successfully.")
    return cert_path, key_path

def get_server_ssl_context(cert_path="server.crt", key_path="server.key"):
    """Creates a TLS Server SSLContext enforcing TLS 1.2 / TLS 1.3."""
    cert_path, key_path = ensure_tls_credentials(cert_path, key_path)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    return ctx

def get_client_ssl_context(ca_cert=None, allow_self_signed=True):
    """Creates a TLS Client SSLContext."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    if ca_cert and os.path.exists(ca_cert):
        ctx.load_verify_locations(cafile=ca_cert)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_REQUIRED
    elif allow_self_signed:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED

    return ctx
