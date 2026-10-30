// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "discord_rpc_handler.h"

namespace DiscordRPCHandler {

#ifdef RYTY_GPU
constexpr const char* RunningOn = nullptr; // Omit the client branding from RyGT presence.
#else
constexpr const char* RunningOn = "Running on shadGT";
#endif

// One Discord application per game: its name is the "Playing" line.
constexpr const char* GtsApplicationId = "1557422166931275906"; // "Gran Turismo Sport"
// Discord's official game id for Gran Turismo 7 (a string: it is above 2^53).
constexpr const char* Gt7ApplicationId = "1440143189372567552"; // "Gran Turismo 7"
// The game's icon as the large image (a public URL, uploaded to the earlier RyGT application).
constexpr const char* Gt7Icon = "https://cdn.discordapp.com/app-icons/1563296318623977472/"
                                "184b3f0882be7369afbcdbb523416da6.png?size=512";

static bool IsGt7(const std::string& game_name, const std::string& game_id) {
    // CUSA24767 (America), CUSA24769 (Europe); other regions by title ("Gran Turismo® 7").
    return game_id == "CUSA24767" || game_id == "CUSA24769" ||
           (game_name.starts_with("Gran Turismo") && game_name.ends_with(" 7"));
}

void RPC::init(const std::string& game_name, const std::string& game_id) {
    gt7 = IsGt7(game_name, game_id) && *Gt7ApplicationId != '\0';
    DiscordEventHandlers handlers{};
    Discord_Initialize(gt7 ? Gt7ApplicationId : GtsApplicationId, &handlers, 1, nullptr);
    rpcEnabled = true;
}

static void SetImage(DiscordRichPresence& rpc, bool gt7) {
    if (gt7) {
        rpc.largeImageKey = Gt7Icon;
        rpc.largeImageText = "Gran Turismo 7";
        return;
    }
    rpc.largeImageKey = "https://raw.githubusercontent.com/mlgprorektm8/shadPS4/main/assets/"
                        "GranTurismoSportIcon.jpg";
    rpc.largeImageText = "Gran Turismo Sport";
}

void RPC::setStatusIdling() {
    DiscordRichPresence rpc{};
    rpc.details = RunningOn;
    SetImage(rpc, gt7);
    status = RPCStatus::Idling;
    Discord_UpdatePresence(&rpc);
}

void RPC::setStatusPlaying(const std::string&, const std::string&) {
    DiscordRichPresence rpc{};
    rpc.details = RunningOn;
    SetImage(rpc, gt7);
    status = RPCStatus::Playing;
    Discord_UpdatePresence(&rpc);
}

void RPC::shutdown() {
    if (rpcEnabled) {
        rpcEnabled = false;
        Discord_ClearPresence();
        Discord_Shutdown();
    }
}

bool RPC::getRPCEnabled() {
    return rpcEnabled;
}

} // namespace DiscordRPCHandler
