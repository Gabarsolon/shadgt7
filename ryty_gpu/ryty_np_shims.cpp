// Stand-ins for the emulator services shadGT's NP libraries call directly. In the Ryty build the
// game's threads, mutexes and memory belong to Ryty's libkernel, so the kernel calls forward to it
// by NID. HTTP forwards to Ryty's libSceHttp and the P2P transport is in ryty_np_p2p.cpp; overlay
// calls do nothing.

#include <mutex>
#include <unordered_map>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>

#include "common/logging/log.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/memory.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/network/http.h"
#include "core/libraries/network/http2.h"
#include "core/libraries/network/net.h"
#include "core/libraries/network/netctl.h"
#include "core/libraries/network/sockets.h"
#include "core/libraries/np/trophy_ui.h"
#include "core/libraries/system/commondialog.h"
#include "core/libraries/system/systemservice.h"
#include "core/libraries/system/userservice.h"
#include "core/memory.h"
#include "imgui/imgui_layer.h"
#include "imgui/invitation_prompt_layer.h"
#include "imgui/shadnet_notifications_layer.h"

namespace {

template <typename T>
T RytyExport(const char* module, const char* nid) {
    HMODULE handle = GetModuleHandleA(module);
    auto* address = handle ? reinterpret_cast<void*>(GetProcAddress(handle, nid)) : nullptr;
    if (address == nullptr) {
        LOG_CRITICAL(Lib_NpManager, "Ryty export {} not found in {}", nid, module);
    }
    return reinterpret_cast<T>(address);
}

// Calls a guest-ABI function that Ryty's libkernel exports under the given NID.
#define RYTY_KERNEL(nid, ...)                                                                      \
    static const auto fn = RytyExport<int(PS4_SYSV_ABI*)(__VA_ARGS__)>("libkernel.prx", nid);     \
    if (fn == nullptr) {                                                                           \
        return ORBIS_KERNEL_ERROR_ENOSYS;                                                          \
    }                                                                                              \
    return fn

constexpr int ErrorOffline = static_cast<int>(0x80410101); // ORBIS_NET_ERROR_ENOTCONN-style failure

} // namespace

namespace Libraries::Kernel {

s32 ErrnoToSceKernelError(s32 error) {
    return error + ORBIS_KERNEL_ERROR_UNKNOWN;
}

int PS4_SYSV_ABI posix_pthread_attr_init(PthreadAttrT* attr) {
    RYTY_KERNEL("wtkt-teR1so", PthreadAttrT*)(attr);
}
int PS4_SYSV_ABI posix_pthread_attr_destroy(PthreadAttrT* attr) {
    RYTY_KERNEL("zHchY8ft5pk", PthreadAttrT*)(attr);
}
int PS4_SYSV_ABI posix_pthread_attr_setstacksize(PthreadAttrT* attr, size_t stacksize) {
    RYTY_KERNEL("2Q0z6rnBrTE", PthreadAttrT*, size_t)(attr, stacksize);
}
int PS4_SYSV_ABI posix_pthread_attr_setinheritsched(PthreadAttrT* attr, int sched_inherit) {
    RYTY_KERNEL("7ZlAakEf0Qg", PthreadAttrT*, int)(attr, sched_inherit);
}
int PS4_SYSV_ABI posix_pthread_attr_setschedpolicy(PthreadAttrT* attr, SchedPolicy policy) {
    RYTY_KERNEL("JarMIy8kKEY", PthreadAttrT*, int)(attr, static_cast<int>(policy));
}
int PS4_SYSV_ABI posix_pthread_attr_setschedparam(PthreadAttrT* attr, SchedParam* param) {
    RYTY_KERNEL("euKRgm0Vn2M", PthreadAttrT*, SchedParam*)(attr, param);
}
int PS4_SYSV_ABI scePthreadAttrSetaffinity(PthreadAttrT* attr, const u64 mask) {
    RYTY_KERNEL("3qxgM4ezETA", PthreadAttrT*, u64)(attr, mask);
}
int PS4_SYSV_ABI posix_pthread_create(PthreadT* thread, const PthreadAttrT* attr,
                                      PthreadEntryFunc start_routine, void* arg) {
    RYTY_KERNEL("OxhIB8LB-PQ", PthreadT*, const PthreadAttrT*, PthreadEntryFunc, void*)(
        thread, attr, start_routine, arg);
}
int PS4_SYSV_ABI posix_pthread_create_name_np(PthreadT* thread, const PthreadAttrT* attr,
                                              PthreadEntryFunc start_routine, void* arg,
                                              const char* name) {
    RYTY_KERNEL("Jmi+9w9u0E4", PthreadT*, const PthreadAttrT*, PthreadEntryFunc, void*,
                const char*)(thread, attr, start_routine, arg, name);
}
int PS4_SYSV_ABI posix_pthread_join(PthreadT pthread, void** thread_return) {
    RYTY_KERNEL("h9CcP3J0oVM", PthreadT, void**)(pthread, thread_return);
}
int PS4_SYSV_ABI posix_pthread_detach(PthreadT pthread) {
    RYTY_KERNEL("+U1R4WtXvoc", PthreadT)(pthread);
}

int PS4_SYSV_ABI posix_pthread_mutexattr_init(PthreadMutexAttrT* attr) {
    RYTY_KERNEL("dQHWEsJtoE4", PthreadMutexAttrT*)(attr);
}
int PS4_SYSV_ABI posix_pthread_mutexattr_settype(PthreadMutexAttrT* attr, PthreadMutexType type) {
    RYTY_KERNEL("mDmgMOGVUqg", PthreadMutexAttrT*, int)(attr, static_cast<int>(type));
}
int PS4_SYSV_ABI posix_pthread_mutexattr_destroy(PthreadMutexAttrT* attr) {
    RYTY_KERNEL("HF7lK46xzjY", PthreadMutexAttrT*)(attr);
}
int PS4_SYSV_ABI scePthreadMutexInit(PthreadMutexT* mutex, const PthreadMutexAttrT* mutex_attr,
                                     const char* name) {
    RYTY_KERNEL("cmo1RIYva9o", PthreadMutexT*, const PthreadMutexAttrT*, const char*)(
        mutex, mutex_attr, name);
}
int PS4_SYSV_ABI posix_pthread_mutex_lock(PthreadMutexT* mutex) {
    RYTY_KERNEL("7H0iTOciTLo", PthreadMutexT*)(mutex);
}
int PS4_SYSV_ABI posix_pthread_mutex_unlock(PthreadMutexT* mutex) {
    RYTY_KERNEL("2Z+PpY6CaJg", PthreadMutexT*)(mutex);
}
int PS4_SYSV_ABI posix_pthread_mutex_destroy(PthreadMutexT* mutex) {
    RYTY_KERNEL("ltCfaGr2JGE", PthreadMutexT*)(mutex);
}
s32 PS4_SYSV_ABI posix_pthread_mutex_trylock(PthreadMutexT* mutex) {
    RYTY_KERNEL("K-jXhbt2gn4", PthreadMutexT*)(mutex);
}

int PS4_SYSV_ABI posix_pthread_condattr_init(PthreadCondAttrT* attr) {
    RYTY_KERNEL("mKoTx03HRWA", PthreadCondAttrT*)(attr);
}
int PS4_SYSV_ABI posix_pthread_condattr_destroy(PthreadCondAttrT* attr) {
    RYTY_KERNEL("dJcuQVn6-Iw", PthreadCondAttrT*)(attr);
}
int PS4_SYSV_ABI scePthreadCondInit(PthreadCondT* cond, const PthreadCondAttrT* cond_attr,
                                    const char* name) {
    RYTY_KERNEL("2Tb92quprl0", PthreadCondT*, const PthreadCondAttrT*, const char*)(
        cond, cond_attr, name);
}
int PS4_SYSV_ABI posix_pthread_cond_destroy(PthreadCondT* cond) {
    RYTY_KERNEL("RXXqi4CtF8w", PthreadCondT*)(cond);
}
int PS4_SYSV_ABI posix_pthread_cond_signal(PthreadCondT* cond) {
    RYTY_KERNEL("2MOy+rUfuhQ", PthreadCondT*)(cond);
}
int PS4_SYSV_ABI posix_pthread_cond_wait(PthreadCondT* cond, PthreadMutexT* mutex) {
    RYTY_KERNEL("Op8TBGY5KHg", PthreadCondT*, PthreadMutexT*)(cond, mutex);
}

s32 PS4_SYSV_ABI sceKernelClockGettime(u32 clock_id, OrbisKernelTimespec* tp) {
    RYTY_KERNEL("QBi7HCK03hw", u32, OrbisKernelTimespec*)(clock_id, tp);
}

// Ryty has no pthread_cond_reltimedwait_np; a default condition variable times out against the
// realtime clock, so the relative wait becomes an absolute one.
int PS4_SYSV_ABI posix_pthread_cond_reltimedwait_np(PthreadCondT* cond, PthreadMutexT* mutex,
                                                    u64 usec) {
    OrbisKernelTimespec now{};
    sceKernelClockGettime(ORBIS_CLOCK_REALTIME, &now);
    const u64 nsec = static_cast<u64>(now.tv_nsec) + (usec % 1000000) * 1000;
    OrbisKernelTimespec abstime{};
    abstime.tv_sec = now.tv_sec + static_cast<s64>(usec / 1000000 + nsec / 1000000000);
    abstime.tv_nsec = static_cast<s64>(nsec % 1000000000);
    RYTY_KERNEL("27bAgiJmOh0", PthreadCondT*, PthreadMutexT*, const OrbisKernelTimespec*)(
        cond, mutex, &abstime);
}

s32 PS4_SYSV_ABI sceKernelUsleep(u32 microseconds) {
    RYTY_KERNEL("1jfXLRVzisc", u32)(microseconds);
}
s32 PS4_SYSV_ABI sceKernelMapNamedFlexibleMemory(void** addr_in_out, u64 len, s32 prot, s32 flags,
                                                 const char* name) {
    RYTY_KERNEL("mL8NDH86iQI", void**, u64, s32, s32, const char*)(addr_in_out, len, prot, flags,
                                                                    name);
}
s32 PS4_SYSV_ABI sceKernelMunmap(void* addr, u64 len) {
    RYTY_KERNEL("cQke9UuBQOk", void*, u64)(addr, len);
}

s32 PS4_SYSV_ABI sceKernelGetSystemSwVersion(SwVersionStruct* ret) {
    if (ret == nullptr) {
        return ORBIS_OK;
    }
    constexpr u32 firmware = 0x05050000;
    ret->hex_representation = firmware;
    std::snprintf(ret->text_representation, sizeof(ret->text_representation), "%2x.%03x.%03x",
                  firmware >> 0x18, firmware >> 0xc & 0xfff, firmware & 0xfff);
    return ORBIS_OK;
}

} // namespace Libraries::Kernel

namespace Libraries::UserService {

s32 PS4_SYSV_ABI sceUserServiceGetInitialUser(int* user_id) {
    static const auto fn = RytyExport<s32(PS4_SYSV_ABI*)(int*)>("libSceUserService.prx",
                                                               "CdWp0oHWGr0");
    return fn ? fn(user_id) : ORBIS_KERNEL_ERROR_ENOSYS;
}

s32 PS4_SYSV_ABI sceUserServiceGetUserName(int user_id, char* user_name, std::size_t size) {
    static const auto fn = RytyExport<s32(PS4_SYSV_ABI*)(int, char*, std::size_t)>(
        "libSceUserService.prx", "1xxcMiGu2fo");
    return fn ? fn(user_id, user_name, size) : ORBIS_KERNEL_ERROR_ENOSYS;
}

} // namespace Libraries::UserService

namespace Libraries::SystemService {
void PushSystemServiceEvent(const OrbisSystemServiceEvent&) {
    LOG_INFO(Lib_NpManager, "Ryty: system service event from the NP libraries dropped");
}
} // namespace Libraries::SystemService

namespace Libraries::CommonDialog {
bool g_isInitialized = false;
bool g_isUsed = false;
} // namespace Libraries::CommonDialog

namespace Libraries::Net {

u16 PS4_SYSV_ABI sceNetHtons(u16 host16) {
    return _byteswap_ushort(host16);
}
u16 PS4_SYSV_ABI sceNetNtohs(u16 net16) {
    return _byteswap_ushort(net16);
}
int PS4_SYSV_ABI sceNetInetPton(int af, const char* src, void* dst) {
    // ORBIS_NET_AF_INET and ORBIS_NET_AF_INET6 match the Windows values for IPv4; IPv6 is 28 there.
    return inet_pton(af == 28 ? AF_INET6 : af, src, dst);
}
// NpSignaling opens a P2P datagram socket at startup only to probe it, and treats a failure as
// fatal (its cleanup then deadlocks). That probe gets a placeholder descriptor, so it never claims
// a virtual port in Ryty's libSceNet (ryty_np_p2p.cpp opens the real signaling socket); other
// sockets are Ryty's.
constexpr OrbisNetId FirstP2PPlaceholder = 0x10000;
std::atomic<OrbisNetId> g_next_p2p_placeholder{FirstP2PPlaceholder};

OrbisNetId PS4_SYSV_ABI sceNetSocket(const char* name, int family, int type, int protocol) {
    if (type == ORBIS_NET_SOCK_DGRAM_P2P || type == ORBIS_NET_SOCK_STREAM_P2P) {
        return g_next_p2p_placeholder++;
    }
    static const auto fn = RytyExport<OrbisNetId(PS4_SYSV_ABI*)(const char*, int, int, int)>(
        "libSceNet.prx", "Q4qBuN-c0ZM");
    return fn ? fn(name, family, type, protocol) : ErrorOffline;
}
int PS4_SYSV_ABI sceNetSocketClose(OrbisNetId s) {
    if (s >= FirstP2PPlaceholder) {
        return ORBIS_OK;
    }
    static const auto fn =
        RytyExport<int(PS4_SYSV_ABI*)(OrbisNetId)>("libSceNet.prx", "45ggEzakPJQ");
    return fn ? fn(s) : ORBIS_OK;
}
int PS4_SYSV_ABI sceNetIoctl() {
    return ORBIS_OK;
}

// The P2P transport for NpSignaling is in ryty_np_p2p.cpp.

} // namespace Libraries::Net

namespace Libraries::NetCtl {
int PS4_SYSV_ABI sceNetCtlGetNatInfo(OrbisNetCtlNatInfo* nat_info) {
    if (nat_info != nullptr) {
        nat_info->stun_status = 0;
        nat_info->nat_type = 0;
        nat_info->mapped_addr = 0;
    }
    return ORBIS_OK;
}
} // namespace Libraries::NetCtl

// The NP web API's HTTP goes through Ryty's libSceHttp, the library the game's own HTTP uses: the
// game hands NpWebApi an HTTP context it made there. Without it the shadNet web API is unreachable.
#define RYTY_HTTP(nid, ...)                                                                        \
    static const auto fn = RytyExport<int(PS4_SYSV_ABI*)(__VA_ARGS__)>("libSceHttp.prx", nid);     \
    if (fn == nullptr) {                                                                           \
        return ErrorOffline;                                                                       \
    }                                                                                              \
    return fn

namespace Libraries::Http {
int PS4_SYSV_ABI sceHttpCreateTemplate(int ctx, const char* agent, int version, int proxy) {
    RYTY_HTTP("0gYjPTR-6cY", int, const char*, int, int)(ctx, agent, version, proxy);
}
int PS4_SYSV_ABI sceHttpCreateConnectionWithURL(int tmpl, const char* url, bool keep_alive) {
    RYTY_HTTP("qgxDBjorUxs", int, const char*, int)(tmpl, url, keep_alive ? 1 : 0);
}
int PS4_SYSV_ABI sceHttpCreateRequestWithURL(int conn, s32 method, const char* url, u64 length) {
    RYTY_HTTP("Aeu5wVKkF9w", int, s32, const char*, u64)(conn, method, url, length);
}
int PS4_SYSV_ABI sceHttpAddRequestHeader(int id, const char* name, const char* value, s32 mode) {
    RYTY_HTTP("EY28T2bkN7k", int, const char*, const char*, u32)(id, name, value,
                                                                 static_cast<u32>(mode));
}
int PS4_SYSV_ABI sceHttpSendRequest(int req, const void* data, u64 size) {
    RYTY_HTTP("1e2BNwI-XzE", int, const void*, u64)(req, data, size);
}
int PS4_SYSV_ABI sceHttpReadData(s32 req, void* data, u64 size) {
    RYTY_HTTP("P5pdoykPYTk", s32, void*, u64)(req, data, size);
}
int PS4_SYSV_ABI sceHttpGetAllResponseHeaders(int req, char** header, u64* size) {
    RYTY_HTTP("aCYPMSUIaP8", int, char**, u64*)(req, header, size);
}
int PS4_SYSV_ABI sceHttpGetStatusCode(int req, int* status) {
    RYTY_HTTP("0a2TBNfE3BU", int, int*)(req, status);
}
int PS4_SYSV_ABI sceHttpParseResponseHeader(const char* header, u64 length, const char* field,
                                            const char** value, u64* value_length) {
    RYTY_HTTP("hPTXo3bICzI", const char*, u64, const char*, const char**, u64*)(
        header, length, field, value, value_length);
}
int PS4_SYSV_ABI sceHttpDeleteConnection(int conn) {
    RYTY_HTTP("P6A3ytpsiYc", int)(conn);
}
int PS4_SYSV_ABI sceHttpDeleteTemplate(int tmpl) {
    RYTY_HTTP("4I8vEpuEhZ8", int)(tmpl);
}
int PS4_SYSV_ABI sceHttpDeleteRequest(int req) {
    RYTY_HTTP("qe7oZ+v4PWA", int)(req);
}
int PS4_SYSV_ABI sceHttpInit(int memid, int ssl_ctx, u64 pool_size) {
    RYTY_HTTP("A9cVMUtEp4Y", int, int, u64)(memid, ssl_ctx, pool_size);
}
int PS4_SYSV_ABI sceHttpCreateRequestWithURL2(int conn, const char* method, const char* url, u64 length) {
    RYTY_HTTP("Cnp77podkCU", int, const char*, const char*, u64)(conn, method, url, length);
}
int PS4_SYSV_ABI sceHttpSetRequestContentLength(int req, u64 length) {
    RYTY_HTTP("PTiFIUxCpJc", int, u64)(req, length);
}
} // namespace Libraries::Http

#undef RYTY_HTTP

// The web API's HTTP/2 goes out through the game's libSceHttp (RyGT's, which sends it to
// RyGTServer) as HTTP/1.1: a template is a libSceHttp template, and a request gets a connection
// of its own.
namespace Libraries::Http2 {
namespace {
std::mutex g_connections_mutex;
std::unordered_map<s32, s32> g_connections; // request -> its connection
} // namespace

s32 PS4_SYSV_ABI sceHttp2CreateTemplate(s32 ctx, const char* agent, s32, s32 proxy) {
    const s32 result = Libraries::Http::sceHttpCreateTemplate(ctx, agent, 2 /* HTTP/1.1 */, proxy);
    LOG_INFO(Lib_NpWebApi2, "HTTP template for context {}: {:#x}", ctx, result);
    return result;
}
// PSN calls shadNet does not answer go to RyGTServer (online_server in RyGT.ini) as
// /rygt/psn/<path>.
std::string RouteUrl(const char* url) {
    static const std::string server = [] {
        std::string value;
        if (std::FILE* ini = std::fopen("RyGT.ini", "r")) {
            char line[256];
            while (std::fgets(line, sizeof(line), ini)) {
                if (std::strncmp(line, "online_server=", 14) == 0) {
                    value = line + 14;
                    value.erase(value.find_last_not_of(" \r\n") + 1);
                }
            }
            std::fclose(ini);
        }
        return value;
    }();
    const std::string full = url;
    const auto scheme = full.find("://");
    const auto slash = scheme == std::string::npos ? std::string::npos : full.find('/', scheme + 3);
    if (server.empty() || slash == std::string::npos) {
        return full;
    }
    const std::string path = full.substr(slash);
    if (path.rfind("/v3/users/me/communication/restriction/status", 0) == 0 ||
        path.rfind("/v2/profanity/filter/", 0) == 0 || path.rfind("/v1/gameSessions", 0) == 0 ||
        path.rfind("/v1/playerSessions", 0) == 0 || path.rfind("/v1/matches", 0) == 0) {
        return "http://" + server + "/rygt/psn" + path;
    }
    return full;
}

s32 PS4_SYSV_ABI sceHttp2CreateRequestWithURL(s32 tmpl, const char* method, const char* original_url, u64 length) {
    const std::string routed = RouteUrl(original_url);
    const char* url = routed.c_str();
    const s32 conn = Libraries::Http::sceHttpCreateConnectionWithURL(tmpl, url, true);
    if (conn < 0) {
        LOG_ERROR(Lib_NpWebApi2, "connection to {} (template {}) failed: {:#x}", url, tmpl, conn);
        return conn;
    }
    const s32 req = Libraries::Http::sceHttpCreateRequestWithURL2(conn, method, url, length);
    if (req < 0) {
        LOG_ERROR(Lib_NpWebApi2, "{} request to {} failed: {:#x}", method, url, req);
        Libraries::Http::sceHttpDeleteConnection(conn);
        return req;
    }
    std::lock_guard lock(g_connections_mutex);
    g_connections[req] = conn;
    return req;
}
s32 PS4_SYSV_ABI sceHttp2AddRequestHeader(s32 req, const char* name, const char* value, u32 mode) {
    return Libraries::Http::sceHttpAddRequestHeader(req, name, value, static_cast<s32>(mode));
}
s32 PS4_SYSV_ABI sceHttp2SetRequestContentLength(s32 req, u64 length) {
    return Libraries::Http::sceHttpSetRequestContentLength(req, length);
}
s32 PS4_SYSV_ABI sceHttp2SetPreSendCallback(s32, OrbisHttp2PreSendCallback, void*) {
    return ORBIS_OK; // shadGT's web API callback does nothing yet
}
s32 PS4_SYSV_ABI sceHttp2SendRequest(s32 req, const void* data, u64 size) {
    return Libraries::Http::sceHttpSendRequest(req, data, size);
}
s32 PS4_SYSV_ABI sceHttp2ReadData(s32 req, void* data, u64 size) {
    return Libraries::Http::sceHttpReadData(req, data, size);
}
s32 PS4_SYSV_ABI sceHttp2GetAllResponseHeaders(s32 req, char** header, u64* size) {
    return Libraries::Http::sceHttpGetAllResponseHeaders(req, header, size);
}
s32 PS4_SYSV_ABI sceHttp2GetStatusCode(s32 req, s32* status) {
    return Libraries::Http::sceHttpGetStatusCode(req, status);
}
s32 PS4_SYSV_ABI sceHttp2AbortRequest(s32) {
    return ORBIS_OK;
}
s32 PS4_SYSV_ABI sceHttp2DeleteRequest(s32 req) {
    const s32 result = Libraries::Http::sceHttpDeleteRequest(req);
    std::lock_guard lock(g_connections_mutex);
    if (const auto it = g_connections.find(req); it != g_connections.end()) {
        Libraries::Http::sceHttpDeleteConnection(it->second);
        g_connections.erase(it);
    }
    return result;
}
s32 PS4_SYSV_ABI sceHttp2DeleteTemplate(s32 tmpl) {
    return Libraries::Http::sceHttpDeleteTemplate(tmpl);
}
} // namespace Libraries::Http2

namespace Libraries::Np::NpTrophy {
void AddTrophyToQueue(const std::filesystem::path&, const std::string& trophyName,
                      const std::string_view& rarity) {
    LOG_INFO(Lib_NpTrophy, "Trophy unlocked: {} ({})", trophyName, rarity);
}
} // namespace Libraries::Np::NpTrophy

// The overlays live in libSceVideoOut.prx's ImGui context, which this library cannot draw into, so
// the commerce dialog's layer is never shown.
namespace ImGui {
void Layer::AddLayer(Layer*) {
    LOG_WARNING(Lib_NpManager, "Ryty: an NP overlay was opened but cannot be drawn");
}
void Layer::RemoveLayer(Layer*) {}
} // namespace ImGui

namespace ImGui::InvitationPrompt {
void Push(s32, std::string, std::string, std::string) {}
void Dismiss(const std::string&) {}
} // namespace ImGui::InvitationPrompt

namespace ImGui::ShadNetNotify {
void Push(Kind, std::string text) {
    LOG_INFO(Lib_NpManager, "shadNet: {}", text);
}
} // namespace ImGui::ShadNetNotify

namespace Core {
void MemoryManager::InvalidateMemory(VAddr, u64) const {}
} // namespace Core
