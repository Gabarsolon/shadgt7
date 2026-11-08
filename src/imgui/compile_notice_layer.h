// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace ImGui::CompileNotice {

/// FIX-GT7-100: while the game shows no new frame for half a second (a load, a draw waiting for
/// its pipeline), the last frame is shown again with a small "Compiling shaders" notice and a
/// spinner, so a long first load doesn't look like a crash. RYGT_COMPILE_NOTICE=0 turns it off.
void Register();
void Unregister();

} // namespace ImGui::CompileNotice
