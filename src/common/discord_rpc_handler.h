// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <discord_rpc.h>

namespace DiscordRPCHandler {

enum class RPCStatus {
    Idling,
    Playing,
};

class RPC {
    bool rpcEnabled = false;
    bool gt7 = false; ///< RyGT: Gran Turismo 7's presence instead of GT Sport's
    RPCStatus status;

public:
    /// RyGT: the game's title and id (its param.sfo) pick the Discord application, as Discord
    /// shows "Playing <application name>".
    void init(const std::string& game_name = {}, const std::string& game_id = {});
    void setStatusIdling();
    void setStatusPlaying(const std::string& game_name, const std::string& game_id);
    void shutdown();
    bool getRPCEnabled();
};

} // namespace DiscordRPCHandler
