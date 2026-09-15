#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <functional>
#include <chrono>
#include <thread>
#include <fstream>
#include <cstring>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET socket_t;
    #define INVALID_SOCK INVALID_SOCKET
    #define SOCK_ERR SOCKET_ERROR
    #define CLOSE_SOCK closesocket
#else
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    typedef int socket_t;
    #define INVALID_SOCK (-1)
    #define SOCK_ERR (-1)
    #define CLOSE_SOCK close
#endif

namespace kawpow::net {

inline void init_network() {
#ifdef _WIN32
    static bool initialized = false;
    if (!initialized) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        initialized = true;
    }
#endif
}

class TcpSocket {
public:
    socket_t sock = INVALID_SOCK;

    TcpSocket() { init_network(); }
    explicit TcpSocket(socket_t s) : sock(s) { init_network(); }
    ~TcpSocket() { close(); }

    TcpSocket(TcpSocket&& other) noexcept : sock(other.sock) { other.sock = INVALID_SOCK; }
    TcpSocket& operator=(TcpSocket&& other) noexcept {
        if (this != &other) {
            close();
            sock = other.sock;
            other.sock = INVALID_SOCK;
        }
        return *this;
    }

    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;

    bool is_valid() const { return sock != INVALID_SOCK; }

    void close() {
        if (sock != INVALID_SOCK) {
            CLOSE_SOCK(sock);
            sock = INVALID_SOCK;
        }
    }

    bool connect(const std::string& host, int port) {
        close();
        struct addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;

        std::string port_str = std::to_string(port);
        if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0 || !res) {
            return false;
        }

        sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (sock == INVALID_SOCK) {
            freeaddrinfo(res);
            return false;
        }

        int nodelay = 1;
        setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));

        if (::connect(sock, res->ai_addr, static_cast<int>(res->ai_addrlen)) == SOCK_ERR) {
            close();
            freeaddrinfo(res);
            return false;
        }

        freeaddrinfo(res);
        return true;
    }

    bool send(const void* data, size_t len) {
        if (!is_valid()) return false;
        const char* ptr = reinterpret_cast<const char*>(data);
        size_t sent = 0;
        while (sent < len) {
            int ret = ::send(sock, ptr + sent, static_cast<int>(len - sent), 0);
            if (ret <= 0) return false;
            sent += ret;
        }
        return true;
    }

    bool send_string(const std::string& str) {
        return send(str.data(), str.size());
    }

    int recv(void* buf, size_t len) {
        if (!is_valid()) return -1;
        return ::recv(sock, reinterpret_cast<char*>(buf), static_cast<int>(len), 0);
    }

    bool readline(std::string& line) {
        line.clear();
        if (!is_valid()) return false;
        char c;
        while (true) {
            int ret = ::recv(sock, &c, 1, 0);
            if (ret <= 0) return false;
            if (c == '\n') break;
            if (c != '\r') line.push_back(c);
        }
        return true;
    }
};

class TcpServer {
public:
    socket_t listen_sock = INVALID_SOCK;

    TcpServer() { init_network(); }
    ~TcpServer() { close(); }

    void close() {
        if (listen_sock != INVALID_SOCK) {
            CLOSE_SOCK(listen_sock);
            listen_sock = INVALID_SOCK;
        }
    }

    bool bind_and_listen(int port, const std::string& ip = "0.0.0.0", int backlog = 64) {
        close();
        listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listen_sock == INVALID_SOCK) {
#ifdef _WIN32
            std::cerr << "[Network] socket() failed, WSAGetLastError=" << WSAGetLastError() << "\n";
#endif
            return false;
        }

        int opt = 1;
        setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (ip.empty() || ip == "0.0.0.0") {
            addr.sin_addr.s_addr = htonl(INADDR_ANY);
        } else {
            inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
        }

        if (::bind(listen_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCK_ERR) {
#ifdef _WIN32
            std::cerr << "[Network] bind() on port " << port << " failed, WSAGetLastError=" << WSAGetLastError() << "\n";
#endif
            close();
            return false;
        }

        if (::listen(listen_sock, backlog) == SOCK_ERR) {
#ifdef _WIN32
            std::cerr << "[Network] listen() failed, WSAGetLastError=" << WSAGetLastError() << "\n";
#endif
            close();
            return false;
        }

        return true;
    }

    TcpSocket accept() {
        if (listen_sock == INVALID_SOCK) return TcpSocket(INVALID_SOCK);
        sockaddr_in client_addr{};
#ifdef _WIN32
        int addr_len = sizeof(client_addr);
#else
        socklen_t addr_len = sizeof(client_addr);
#endif
        socket_t client_sock = ::accept(listen_sock, reinterpret_cast<sockaddr*>(&client_addr), &addr_len);
        if (client_sock == INVALID_SOCK) return TcpSocket(INVALID_SOCK);

        int nodelay = 1;
        setsockopt(client_sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));

        return TcpSocket(client_sock);
    }
};

} // namespace kawpow::net
