// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>
#include "common/perf_monitor.h"
#include "common/small_vector.h"
#include "common/types.h"
#include "shader_recompiler/info.h"

namespace Serialization {
struct Archive;
}

namespace Shader::Gcn {

struct VertexAttribute {
    enum InstanceIdType : u8 {
        None = 0,
        OverStepRate0 = 1,
        OverStepRate1 = 2,
        Plain = 3,
    };

    u8 dest_vgpr;     ///< Destination VGPR to load first component.
    u8 num_elements;  ///< Number of components to load
    u8 sgpr_base;     ///< SGPR that contains the pointer to the list of vertex V#
    u8 dword_offset;  ///< The dword offset of the V# that describes this attribute.
    u8 instance_data; ///< Indicates that the buffer will be accessed in instance rate
    u8 inst_offset;   ///< Instruction offset applied on the formatted buffer loads
    u8 data_format;   ///< Data format override when typed buffer loads are used
    u8 num_format;    ///< Number format override when typed buffer loads are used

    InstanceIdType GetStepRate() const {
        return static_cast<InstanceIdType>(instance_data);
    }

    AmdGpu::Buffer GetSharp(const Shader::Info& info) const noexcept {
        auto buffer = info.ReadFetchSharp(sgpr_base, dword_offset);
        buffer.base_address += inst_offset;
        if (data_format) {
            buffer.data_format = data_format;
            buffer.num_format = num_format;
        }
        return buffer;
    }

    /// FIX-060: a V# of a gigabyte or more is garbage, and counts as plain floats in the key.
    static bool IsGarbage(const AmdGpu::Buffer& sharp) {
        static const bool enabled = Common::PerfFeatureEnabled(83);
        return enabled && sharp.GetSize() >= (u64{1} << 30);
    }

    /// FIX-GT7-054: the V# code is made from. Code made from a garbage V# followed its swizzle
    /// under the key of plain floats, and a real attribute of four floats then matched it: GT7's
    /// map markers read R, G, B, R or nothing at all. RYGT_GARBAGE_VSHARP_PLAIN=0: as before.
    AmdGpu::Buffer GetSharpForCode(const Shader::Info& info) const noexcept {
        static const bool plain = [] {
            const char* env = std::getenv("RYGT_GARBAGE_VSHARP_PLAIN");
            return !(env && env[0] == '0');
        }();
        auto buffer = GetSharp(info);
        if (plain && IsGarbage(buffer)) {
            buffer.data_format = static_cast<u32>(AmdGpu::DataFormat::Format32_32_32_32);
            buffer.num_format = static_cast<u32>(AmdGpu::NumberFormat::Float);
            buffer.dst_sel_x = static_cast<u32>(AmdGpu::CompSwizzle::Red);
            buffer.dst_sel_y = static_cast<u32>(AmdGpu::CompSwizzle::Green);
            buffer.dst_sel_z = static_cast<u32>(AmdGpu::CompSwizzle::Blue);
            buffer.dst_sel_w = static_cast<u32>(AmdGpu::CompSwizzle::Alpha);
        }
        return buffer;
    }

    bool operator==(const VertexAttribute& other) const = default;
};

struct FetchShaderData {
    u32 size = 0;
    SmallVector<VertexAttribute, 8> attributes;
    s8 vertex_offset_sgpr = -1;   ///< SGPR of vertex offset from VADDR
    s8 instance_offset_sgpr = -1; ///< SGPR of instance offset from VADDR

    bool operator==(const FetchShaderData& other) const {
        return attributes == other.attributes && vertex_offset_sgpr == other.vertex_offset_sgpr &&
               instance_offset_sgpr == other.instance_offset_sgpr;
    }

    bool Empty() const {
        return size == 0;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& buffer);
};

const u32* GetFetchShaderCode(const Info& info, u32 sgpr_base);

bool ParseFetchShader(const Shader::Info& info, FetchShaderData& out_fetch_data);

} // namespace Shader::Gcn
