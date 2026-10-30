// PSN signaling for Ryty: how NpSignaling reaches other players.
//
// shadPS4's NpSignaling runs a handshake with each peer before the game talks to it, over a
// transport the network library provides, and asks a resolver where a peer is. In the Ryty build:
//   - the transport is a P2P socket in Ryty's libSceNet bound to its raw virtual port, so
//     handshakes leave from the game's own P2P port (UDP 3658) and the address a peer learns from
//     them is the one the game's own P2P traffic will use;
//   - the resolver is RyGTServer: each game registers its player there (online ID, account ID and
//     P2P port) and looks other players up by online ID or by PSN account ID, which is what GT
//     Sport's lobby uses (sceNpSignalingActivateConnectionA).
// shadNet's own resolver and STUN echo are replaced: for a game on the same PC as shadNet they
// would advertise 127.0.0.1, which other players cannot reach.

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <cctype>
#include <optional>
#include <mutex>
#include <string>
#include <vector>

#include "common/logging/log.h"
#include "core/libraries/network/net.h"
#include "core/libraries/network/sockets.h"
#include "core/libraries/np/np_signaling/np_signaling_state.h"
#include "core/libraries/np/np_signaling/np_signaling_stubs.h"
#include "core/libraries/np/np_types.h"
#include "ryty_np_p2p.h"

namespace {

constexpr int NetSockDgramP2P = 6;
constexpr int NetSolSocket = 0xFFFF;
constexpr int NetSoNbio = 0x1200;
constexpr u16 RawVport = 0xFFFF;  // libSceNet's raw P2P virtual port
constexpr u16 DefaultP2PPort = 3658;

template <typename T>
T NetExport(const char* nid) {
    HMODULE handle = GetModuleHandleA("libSceNet.prx");
    auto* address = handle ? reinterpret_cast<void*>(GetProcAddress(handle, nid)) : nullptr;
    if (address == nullptr) {
        LOG_CRITICAL(Lib_NpSignaling, "Ryty libSceNet export {} not found", nid);
    }
    return reinterpret_cast<T>(address);
}

struct GuestSockaddr {
    u8 len = 16;
    u8 family = 2;
    u16 port = 0;  // network order
    u32 addr = 0;  // network order
    u16 vport = 0; // network order
    u8 zero[6]{};
};

std::mutex g_mutex;
int g_socket = -1;
struct Datagram {
    std::vector<u8> data;
    u32 addr;
    u16 port;
};
std::deque<Datagram> g_signaling;
std::deque<Datagram> g_control;

u16 P2PPort() {
    if (const char* text = std::getenv("RYGT_P2P_PORT")) {
        const long value = std::strtol(text, nullptr, 10);
        if (value > 0 && value < 65536) return static_cast<u16>(value);
    }
    return DefaultP2PPort;
}

std::string Trim(std::string text) {
    const auto start = text.find_first_not_of(" \t\r\n");
    const auto end = text.find_last_not_of(" \t\r\n");
    return start == std::string::npos ? std::string() : text.substr(start, end - start + 1);
}

// RyGTServer's address: RYGT_ONLINE_SERVER, else online_server in RyGT.ini, else this PC.
std::string ServerAddress() {
    if (const char* text = std::getenv("RYGT_ONLINE_SERVER"); text && *text) return text;
    std::ifstream ini("RyGT.ini");
    std::string line;
    while (std::getline(ini, line)) {
        const auto eq = line.find('=');
        if (eq != std::string::npos && Trim(line.substr(0, eq)) == "online_server") {
            const std::string value = Trim(line.substr(eq + 1));
            if (!value.empty()) return value;
        }
    }
    return "127.0.0.1:7890";
}

// A small blocking HTTP GET to RyGTServer; returns the body of a 200 reply.
bool HttpGet(const std::string& path, std::string* body) {
    static const int started = [] {
        WSADATA data{};
        return WSAStartup(MAKEWORD(2, 2), &data);
    }();
    (void)started;
    std::string host = ServerAddress();
    std::string port = "7890";
    const auto colon = host.rfind(':');
    if (colon != std::string::npos) {
        port = host.substr(colon + 1);
        host = host.substr(0, colon);
    }
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &found) != 0 || !found) return false;
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    const DWORD timeout_ms = 3000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
    const bool connected = s != INVALID_SOCKET && connect(s, found->ai_addr, static_cast<int>(found->ai_addrlen)) == 0;
    freeaddrinfo(found);
    if (!connected) {
        if (s != INVALID_SOCKET) closesocket(s);
        return false;
    }
    const std::string request = "GET " + path + " HTTP/1.1\r\nHost: rygt\r\nConnection: close\r\n\r\n";
    send(s, request.data(), static_cast<int>(request.size()), 0);
    std::string reply;
    char buffer[2048];
    for (int got; (got = recv(s, buffer, sizeof(buffer), 0)) > 0;) reply.append(buffer, got);
    closesocket(s);
    if (reply.compare(0, 12, "HTTP/1.1 200") != 0) return false;
    const auto split = reply.find("\r\n\r\n");
    if (body) *body = split == std::string::npos ? std::string() : reply.substr(split + 4);
    return true;
}

std::string JsonField(const std::string& json, const char* key) {
    const std::string pattern = std::string("\"") + key + "\":";
    auto at = json.find(pattern);
    if (at == std::string::npos) return {};
    at += pattern.size();
    if (at < json.size() && json[at] == '"') {
        const auto end = json.find('"', at + 1);
        return end == std::string::npos ? std::string() : json.substr(at + 1, end - at - 1);
    }
    const auto end = json.find_first_of(",}", at);
    return json.substr(at, end == std::string::npos ? std::string::npos : end - at);
}

std::string UrlEncode(std::string_view text) {
    std::string out;
    char hex[4];
    for (unsigned char c : text) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.') {
            out += static_cast<char>(c);
        } else {
            std::snprintf(hex, sizeof(hex), "%%%02X", c);
            out += hex;
        }
    }
    return out;
}

// One peer lookup: ?online=<id> or ?account=<n>.
// Which online ID each PSN account is, from every lookup (both directions), so the signaling calls
// that run under its lock can answer without asking RyGTServer.
std::mutex g_names_mutex;
std::map<u64, std::string> g_name_of_account;
std::map<std::string, u64> g_account_of_name;  // online IDs in lower case

std::string Lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

bool LookupPeer(const std::string& query, std::string* online_id, u32* addr_nbo, u16* port_nbo) {
    std::string body;
    if (!HttpGet("/rygt/p2p/peer?" + query, &body)) return false;
    {
        const std::string name = JsonField(body, "onlineId");
        const u64 account = std::strtoull(JsonField(body, "accountId").c_str(), nullptr, 10);
        if (!name.empty() && account != 0) {
            std::lock_guard lock(g_names_mutex);
            g_name_of_account[account] = name;
            g_account_of_name[Lower(name)] = account;
        }
    }
    const std::string address = JsonField(body, "address");
    const int port = std::atoi(JsonField(body, "port").c_str());
    in_addr parsed{};
    if (address.empty() || port <= 0 || inet_pton(AF_INET, address.c_str(), &parsed) != 1) return false;
    if (online_id) *online_id = JsonField(body, "onlineId");
    if (addr_nbo) *addr_nbo = parsed.s_addr;
    if (port_nbo) *port_nbo = htons(static_cast<u16>(port));
    return true;
}

// Moves every waiting datagram off the raw socket into the signaling or control queue.
void Pump() {
    using RecvFrom = s64(PS4_SYSV_ABI*)(int, void*, u64, int, void*, u32*);
    static const auto recvfrom_fn = NetExport<RecvFrom>("304ooNZxWDY");
    if (g_socket < 0 || !recvfrom_fn) return;
    u8 buffer[2048];
    for (;;) {
        GuestSockaddr from{};
        u32 from_len = sizeof(from);
        const s64 got = recvfrom_fn(g_socket, buffer, sizeof(buffer), 0, &from, &from_len);
        if (got < 0) return;
        Datagram datagram{std::vector<u8>(buffer, buffer + got), from.addr, from.port};
        const bool control = got >= 5 && std::memcmp(buffer, Libraries::Np::NpSignaling::kSignalingMagic, 4) == 0 &&
                             buffer[4] == static_cast<u8>(Libraries::Np::NpSignaling::SignalingPacketType::Control);
        (control ? g_control : g_signaling).push_back(std::move(datagram));
        if (g_signaling.size() > 256) g_signaling.pop_front();
        if (g_control.size() > 256) g_control.pop_front();
    }
}

int Pop(std::deque<Datagram>& queue, void* buf, u32 len, u32* from_addr, u16* from_port) {
    std::lock_guard lock(g_mutex);
    Pump();
    if (queue.empty()) return -1;
    Datagram datagram = std::move(queue.front());
    queue.pop_front();
    const u32 n = std::min<u32>(len, static_cast<u32>(datagram.data.size()));
    std::memcpy(buf, datagram.data.data(), n);
    if (from_addr) *from_addr = datagram.addr;
    if (from_port) *from_port = datagram.port;
    return static_cast<int>(n);
}

int SendRaw(const void* data, u32 len, u32 dest_addr, u16 dest_port) {
    using SendTo = s64(PS4_SYSV_ABI*)(int, const void*, u64, int, const void*, u32);
    static const auto sendto_fn = NetExport<SendTo>("gvD1greCu0A");
    if (!Libraries::Net::EnsureP2PTransport() || !sendto_fn) return -1;
    GuestSockaddr to{};
    to.addr = dest_addr;
    to.port = dest_port;
    to.vport = htons(RawVport);
    return static_cast<int>(sendto_fn(g_socket, data, len, 0, &to, sizeof(to)));
}

bool ResolveByOnlineId(std::string_view online_id, u32* out_addr, u16* out_port) {
    const bool found = LookupPeer("online=" + UrlEncode(online_id), nullptr, out_addr, out_port);
    LOG_INFO(Lib_NpSignaling, "RyGTServer lookup of '{}': {}", online_id, found ? "found" : "not registered");
    return found;
}

} // namespace

namespace Libraries::Net {

u16 GetP2PConfiguredPort() {
    return htons(P2PPort());
}

u32 GetP2PAdvertisedAddr() {
    return 0;
}

bool EnsureP2PTransport() {
    std::lock_guard lock(g_mutex);
    if (g_socket >= 0) return true;
    using Socket = int(PS4_SYSV_ABI*)(const char*, int, int, int);
    using Bind = int(PS4_SYSV_ABI*)(int, const void*, u32);
    using SetSockOpt = int(PS4_SYSV_ABI*)(int, int, int, const void*, u32);
    using Close = int(PS4_SYSV_ABI*)(int);
    static const auto socket_fn = NetExport<Socket>("Q4qBuN-c0ZM");
    static const auto bind_fn = NetExport<Bind>("bErx49PgxyY");
    static const auto setsockopt_fn = NetExport<SetSockOpt>("2mKX2Spso7I");
    static const auto close_fn = NetExport<Close>("45ggEzakPJQ");
    if (!socket_fn || !bind_fn || !setsockopt_fn) return false;
    const int s = socket_fn("RyGTSignaling", 2, NetSockDgramP2P, 0);
    if (s < 0) return false;
    const int on = 1;
    setsockopt_fn(s, NetSolSocket, NetSoNbio, &on, sizeof(on));
    GuestSockaddr local{};
    local.port = htons(P2PPort());
    local.vport = htons(RawVport);
    if (bind_fn(s, &local, sizeof(local)) < 0) {
        LOG_ERROR(Lib_NpSignaling, "cannot bind the signaling socket to P2P port {}", P2PPort());
        if (close_fn) close_fn(s);
        return false;
    }
    g_socket = s;
    LOG_INFO(Lib_NpSignaling, "signaling on P2P port {}", P2PPort());
    return true;
}

bool P2PTransportIsReady() {
    std::lock_guard lock(g_mutex);
    return g_socket >= 0;
}

int P2PSignalingSendTo(const void* data, u32 len, u32 dest_addr, u16 dest_port) {
    return SendRaw(data, len, dest_addr, dest_port);
}

int P2PSignalingRecvFrom(void* buf, u32 len, u32* from_addr, u16* from_port) {
    return Pop(g_signaling, buf, len, from_addr, from_port);
}

int P2PControlSendTo(const void* data, u32 len, u32 dest_addr, u16 dest_port) {
    return SendRaw(data, len, dest_addr, dest_port);
}

int P2PControlRecvFrom(void* buf, u32 len, u32* from_addr, u16* from_port) {
    return Pop(g_control, buf, len, from_addr, from_port);
}

int P2PMatching2SendTo(const void*, u32, u32, u16) {
    return -1;
}

int P2PMatching2RecvFrom(void*, u32, u32*, u16*) {
    return -1;
}

} // namespace Libraries::Net

namespace Ryty {

void InstallSignaling() {
    namespace Stubs = Libraries::Np::NpSignaling::Stubs;
    Stubs::SetTransportHooks({
        .signaling_send = Libraries::Net::P2PSignalingSendTo,
        .signaling_recv = Libraries::Net::P2PSignalingRecvFrom,
        .control_send = Libraries::Net::P2PControlSendTo,
        .control_recv = Libraries::Net::P2PControlRecvFrom,
        .transport_ready = Libraries::Net::P2PTransportIsReady,
        .configured_port = Libraries::Net::GetP2PConfiguredPort,
        .advertised_addr = Libraries::Net::GetP2PAdvertisedAddr,
        .ensure_transport = Libraries::Net::EnsureP2PTransport,
    });
    Stubs::SetPeerResolver(ResolveByOnlineId);
    // No STUN through shadNet: RyGTServer hands out addresses (see the top of this file).
    Stubs::SetMatching2Enabled(false);
    Stubs::SetMmServerEndpoint(0, 0);
    LOG_INFO(Lib_NpSignaling, "peers resolved through RyGTServer at {}", ServerAddress());
}

void RegisterPlayer(std::string_view online_id, u64 account_id) {
    const std::string path = "/rygt/p2p/register?online=" + UrlEncode(online_id) +
                             "&account=" + std::to_string(account_id) + "&port=" + std::to_string(P2PPort());
    if (HttpGet(path, nullptr)) {
        LOG_INFO(Lib_NpSignaling, "registered '{}' (account {}) with RyGTServer, P2P port {}", online_id, account_id,
                 P2PPort());
    } else {
        LOG_WARNING(Lib_NpSignaling, "could not register '{}' with RyGTServer at {}", online_id, ServerAddress());
    }
}

bool OnlineIdForAccount(u64 account_id, std::string* online_id) {
    {
        std::lock_guard lock(g_names_mutex);
        const auto it = g_name_of_account.find(account_id);
        if (it != g_name_of_account.end()) {
            *online_id = it->second;
            return true;
        }
    }
    return LookupPeer("account=" + std::to_string(account_id), online_id, nullptr, nullptr) && !online_id->empty();
}

bool AccountForOnlineId(std::string_view online_id, u64* account_id) {
    {
        std::lock_guard lock(g_names_mutex);
        const auto it = g_account_of_name.find(Lower(online_id));
        if (it != g_account_of_name.end()) {
            *account_id = it->second;
            return true;
        }
    }
    if (!LookupPeer("online=" + UrlEncode(online_id), nullptr, nullptr, nullptr)) return false;
    std::lock_guard lock(g_names_mutex);
    const auto it = g_account_of_name.find(Lower(online_id));
    if (it == g_account_of_name.end()) return false;
    *account_id = it->second;
    return true;
}

} // namespace Ryty
