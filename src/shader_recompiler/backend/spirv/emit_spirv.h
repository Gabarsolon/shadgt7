// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Shader::Backend::SPIRV {

/// FIX-226: float atomic min/max without VK_EXT_shader_atomic_float2 run only the integer atomic
/// the value's sign picks (RYGT_FLOAT_ATOMIC_EXACT=0: both, as before).
bool FloatAtomicExact();

[[nodiscard]] std::vector<u32> EmitSPIRV(const Profile& profile, const RuntimeInfo& runtime_info,
                                         const IR::Program& program, Bindings& binding);

} // namespace Shader::Backend::SPIRV
