// PSN signaling for Ryty (see ryty_np_p2p.cpp).
#pragma once

#include <string>
#include <string_view>

#include "common/types.h"

namespace Ryty {

// Points NpSignaling at the P2P transport and at RyGTServer for peer addresses. Called after the
// shadNet client is set up, which installs shadNet's own resolver.
void InstallSignaling();

// Tells RyGTServer where this player's game takes P2P traffic.
void RegisterPlayer(std::string_view online_id, u64 account_id);

// The online ID RyGTServer has for a PSN account ID, for sceNpSignalingActivateConnectionA.
bool OnlineIdForAccount(u64 account_id, std::string* online_id);

// The PSN account ID RyGTServer has for an online ID (both are remembered after the first lookup).
bool AccountForOnlineId(std::string_view online_id, u64* account_id);

} // namespace Ryty
