// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <array>
#include <boost/container/static_vector.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "common/assert.h"
#include "common/ryty_config.h"
#include "common/debug.h"
#include "common/perf_monitor.h"
#include "common/types.h"
#include "core/emulator_settings.h"
#include "imgui/renderer/imgui_core.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/single_sample_barycentric.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#ifdef RYTY_GPU
#include "ryty_gpu/ryty_dlss.h"
#endif

#include <vk_mem_alloc.h>

namespace Vulkan {

namespace {

std::vector<vk::PhysicalDevice> EnumeratePhysicalDevices(vk::UniqueInstance& instance) {
    auto [devices_result, devices] = instance->enumeratePhysicalDevices();
    ASSERT_MSG(devices_result == vk::Result::eSuccess, "Failed to enumerate physical devices: {}",
               vk::to_string(devices_result));
    return std::move(devices);
}

std::vector<std::string> GetSupportedExtensions(vk::PhysicalDevice physical) {
    const auto [extensions_result, extensions] = physical.enumerateDeviceExtensionProperties();
    if (extensions_result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Could not query supported extensions: {}",
                  vk::to_string(extensions_result));
        return {};
    }
    std::vector<std::string> supported_extensions;
    supported_extensions.reserve(extensions.size());
    for (const auto& extension : extensions) {
        supported_extensions.emplace_back(extension.extensionName.data());
    }
    return supported_extensions;
}

vk::FormatProperties3 GetFormatProperties(vk::PhysicalDevice physical, vk::Format format) {
    vk::FormatProperties3 properties3{};
    vk::FormatProperties2 properties2 = {
        .pNext = &properties3,
    };
    physical.getFormatProperties2(format, &properties2);
    return properties3;
}

std::unordered_map<vk::Format, vk::FormatProperties3> GetFormatProperties(
    vk::PhysicalDevice physical) {
    std::unordered_map<vk::Format, vk::FormatProperties3> format_properties;
    for (const auto& format_info : LiverpoolToVK::SurfaceFormats()) {
        const auto format = format_info.vk_format;
        if (!format_properties.contains(format)) {
            format_properties.emplace(format, GetFormatProperties(physical, format));
        }
    }
    for (const auto& format_info : LiverpoolToVK::DepthFormats()) {
        const auto format = format_info.vk_format;
        if (!format_properties.contains(format)) {
            format_properties.emplace(format, GetFormatProperties(physical, format));
        }
    }
    // Other miscellaneous formats, e.g. for color buffers, swizzles, or compatibility
    static constexpr std::array misc_formats = {
        vk::Format::eA2R10G10B10UnormPack32,
        vk::Format::eB8G8R8A8Unorm,
        vk::Format::eB8G8R8A8Srgb,
        vk::Format::eD24UnormS8Uint,
    };
    for (const auto& format : misc_formats) {
        if (!format_properties.contains(format)) {
            format_properties.emplace(format, GetFormatProperties(physical, format));
        }
    }
    return format_properties;
}

std::string GetReadableVersion(u32 version) {
    return fmt::format("{}.{}.{}", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
                       VK_VERSION_PATCH(version));
}

// AMD-300 (SHADGT_SIMULATE_GPU=gcn|rdna1|rdna2): the device reports only what an older Radeon's
// Windows driver offers (gpuinfo.org: RX 580/RX Vega 2.0.283, RX 5700 XT/RX 6800 2.0.353), so a
// newer card exercises the paths those cards take. It cannot change the hardware's wave size.
enum class SimulatedGpu { None, Gcn, Rdna1, Rdna2 };

SimulatedGpu GetSimulatedGpu() {
    static const SimulatedGpu simulated = [] {
        const char* env = std::getenv("SHADGT_SIMULATE_GPU");
        const std::string_view name = env ? env : "";
        if (name == "gcn" || name == "polaris" || name == "vega") {
            return SimulatedGpu::Gcn;
        }
        if (name == "rdna1") {
            return SimulatedGpu::Rdna1;
        }
        if (name == "rdna2") {
            return SimulatedGpu::Rdna2;
        }
        return SimulatedGpu::None;
    }();
    return simulated;
}

/// AMD-300: SHADGT_SIMULATE_SKIP=<item>,... keeps items real: extension names, f16, gpl, v3d,
/// af2, sub, push.
bool Simulates(std::string_view item) {
    static const std::string skip = [] {
        const char* env = std::getenv("SHADGT_SIMULATE_SKIP");
        return std::string{env ? env : ""};
    }();
    for (size_t pos = 0; pos <= skip.size();) {
        const size_t end = std::min(skip.find(',', pos), skip.size());
        if (std::string_view{skip}.substr(pos, end - pos) == item) {
            return false;
        }
        pos = end + 1;
    }
    return true;
}

void HideSimulatedExtensions(std::vector<std::string>& extensions) {
    std::vector<std::string_view> hidden;
    switch (GetSimulatedGpu()) {
    case SimulatedGpu::Gcn:
        hidden = {VK_KHR_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME,
                  VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME,
                  VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME,
                  VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME,
                  VK_EXT_IMAGE_2D_VIEW_OF_3D_EXTENSION_NAME,
                  VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
                  VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME,
                  VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME};
        break;
    case SimulatedGpu::Rdna1:
        hidden = {VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME};
        break;
    default:
        return;
    }
    std::erase_if(extensions, [&](const std::string& name) {
        const bool driver_side = name == VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME ||
                                 name == VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME;
        return std::ranges::find(hidden, std::string_view{name}) != hidden.end() &&
               Simulates(name) && (!driver_side || Simulates("driver"));
    });
}

} // Anonymous namespace

Instance::Instance(bool enable_validation, bool enable_crash_diagnostic)
    : instance{CreateInstance(Frontend::WindowSystemType::Headless, enable_validation,
                              enable_crash_diagnostic)},
      physical_devices{EnumeratePhysicalDevices(instance)} {}

Instance::Instance(Frontend::WindowSDL& window, s32 physical_device_index,
                   bool enable_validation /*= false*/, bool enable_crash_diagnostic /*= false*/)
    : instance{
          CreateInstance(window.GetWindowInfo().type, enable_validation, enable_crash_diagnostic)},
      physical_devices{EnumeratePhysicalDevices(instance)} {
    if (enable_validation) {
        debug_callback = CreateDebugCallback(*instance);
    }
    const std::size_t num_physical_devices = static_cast<u16>(physical_devices.size());
    ASSERT_MSG(num_physical_devices > 0, "No physical devices found");
    LOG_INFO(Render_Vulkan, "Found {} physical devices", num_physical_devices);

    if (physical_device_index < 0) {
        std::vector<
            std::tuple<size_t, vk::PhysicalDeviceProperties2, vk::PhysicalDeviceMemoryProperties>>
            properties2{};
        for (auto const& physical : physical_devices) {
            properties2.emplace_back(properties2.size(), physical.getProperties2(),
                                     physical.getMemoryProperties());
        }
        std::sort(properties2.begin(), properties2.end(), [](const auto& left, const auto& right) {
            const vk::PhysicalDeviceProperties& left_prop = std::get<1>(left).properties;
            const vk::PhysicalDeviceProperties& right_prop = std::get<1>(right).properties;
            const bool left_supports_api = left_prop.apiVersion >= TargetVulkanApiVersion;
            const bool right_supports_api = right_prop.apiVersion >= TargetVulkanApiVersion;
            if (left_supports_api != right_supports_api) {
                return left_supports_api;
            }

            const bool left_is_discrete =
                left_prop.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
            const bool right_is_discrete =
                right_prop.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
            if (left_is_discrete != right_is_discrete) {
                return left_is_discrete;
            }

            // Software renderers advertise system memory as device local, so they
            // would win the memory comparison below against real hardware.
            const bool left_is_cpu = left_prop.deviceType == vk::PhysicalDeviceType::eCpu;
            const bool right_is_cpu = right_prop.deviceType == vk::PhysicalDeviceType::eCpu;
            if (left_is_cpu != right_is_cpu) {
                return right_is_cpu;
            }

            constexpr auto get_mem = [](const vk::PhysicalDeviceMemoryProperties& mem) -> size_t {
                size_t max = 0;
                for (u32 i = 0; i < mem.memoryHeapCount; i++) {
                    const vk::MemoryHeap& heap = mem.memoryHeaps[i];
                    if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal && heap.size > max) {
                        max = heap.size;
                    }
                }
                return max;
            };
            size_t left_mem_size = get_mem(std::get<2>(left));
            size_t right_mem_size = get_mem(std::get<2>(right));
            return left_mem_size > right_mem_size;
        });
        physical_device = physical_devices[std::get<0>(properties2[0])];
    } else {
        ASSERT_MSG(physical_device_index < num_physical_devices,
                   "Invalid physical device index {} provided when only {} devices exist",
                   physical_device_index, num_physical_devices);

        physical_device = physical_devices[physical_device_index];
    }

    available_extensions = GetSupportedExtensions(physical_device);
    HideSimulatedExtensions(available_extensions); // AMD-300
    format_properties = GetFormatProperties(physical_device);
    properties = physical_device.getProperties();
    memory_properties = physical_device.getMemoryProperties();
    CollectDeviceParameters();
    ASSERT_MSG(properties.apiVersion >= TargetVulkanApiVersion,
               "Vulkan {}.{} is required, but only {}.{} is supported by device!",
               VK_VERSION_MAJOR(TargetVulkanApiVersion), VK_VERSION_MINOR(TargetVulkanApiVersion),
               VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion));

    CreateDevice();
    CollectPhysicalMemoryInfo();
    ApplyVideoMemorySetting();
    CollectImageFormatInfo();
    CollectToolingInfo();
}

Instance::~Instance() {
    ImGui::Core::Shutdown(GetDevice());
    vmaDestroyAllocator(allocator);
}

std::string Instance::GetDriverVersionName() {
    // Extracted from
    // https://github.com/SaschaWillems/vulkan.gpuinfo.org/blob/5dddea46ea1120b0df14eef8f15ff8e318e35462/functions.php#L308-L314
    const u32 version = properties.driverVersion;
    if (driver_id == vk::DriverId::eNvidiaProprietary) {
        const u32 major = (version >> 22) & 0x3ff;
        const u32 minor = (version >> 14) & 0x0ff;
        const u32 secondary = (version >> 6) & 0x0ff;
        const u32 tertiary = version & 0x003f;
        return fmt::format("{}.{}.{}.{}", major, minor, secondary, tertiary);
    }
    if (driver_id == vk::DriverId::eIntelProprietaryWindows) {
        const u32 major = version >> 14;
        const u32 minor = version & 0x3fff;
        return fmt::format("{}.{}", major, minor);
    }
    return GetReadableVersion(version);
}

bool Instance::CreateDevice() {
    vk::StructureChain feature_chain = physical_device.getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceRobustness2FeaturesEXT,
        vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT,
        vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT,
        vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT,
        vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
        vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT, vk::PhysicalDeviceShaderClockFeaturesKHR,
        vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT,
        vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR,
        vk::PhysicalDeviceShaderQuadControlFeaturesKHR>();
    features = feature_chain.get().features;
    // AMD-300: the features an older Radeon's driver turns off.
    if (const auto simulated = GetSimulatedGpu(); simulated != SimulatedGpu::None) {
        auto& vk11 = feature_chain.get<vk::PhysicalDeviceVulkan11Features>();
        auto& vk12 = feature_chain.get<vk::PhysicalDeviceVulkan12Features>();
        vk11.storagePushConstant16 = false;
        if (simulated == SimulatedGpu::Gcn) {
            vk12.shaderFloat16 = vk12.shaderFloat16 && !Simulates("f16");
            if (Simulates("gpl")) {
                feature_chain.get<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>()
                    .graphicsPipelineLibrary = false;
            }
            if (Simulates("v3d")) {
                auto& view_of_3d = feature_chain.get<vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT>();
                view_of_3d.image2DViewOf3D = false;
                view_of_3d.sampler2DViewOf3D = false;
            }
            if (Simulates("af2")) {
                auto& atomic_float2 =
                    feature_chain.get<vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>();
                atomic_float2.shaderBufferFloat32AtomicMinMax = false;
                atomic_float2.shaderSharedFloat32AtomicMinMax = false;
                atomic_float2.shaderImageFloat32AtomicMinMax = false;
            }
        }
        LOG_WARNING(Render_Vulkan, "AMD-300: simulating an older Radeon ({})",
                    simulated == SimulatedGpu::Gcn     ? "GCN: RX 400/500, Vega"
                    : simulated == SimulatedGpu::Rdna1 ? "RDNA 1: RX 5000"
                                                       : "RDNA 2: RX 6000");
    }

    const vk::StructureChain properties_chain = physical_device.getProperties2<
        vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan11Properties,
        vk::PhysicalDeviceVulkan12Properties, vk::PhysicalDeviceVulkan13Properties,
        vk::PhysicalDevicePushDescriptorPropertiesKHR>();
    vk11_props = properties_chain.get<vk::PhysicalDeviceVulkan11Properties>();
    vk12_props = properties_chain.get<vk::PhysicalDeviceVulkan12Properties>();
    vk13_props = properties_chain.get<vk::PhysicalDeviceVulkan13Properties>();
    push_descriptor_props = properties_chain.get<vk::PhysicalDevicePushDescriptorPropertiesKHR>();
    if (const auto simulated = GetSimulatedGpu(); simulated != SimulatedGpu::None) {
        // AMD-300: before RDNA 3 only compute takes a required subgroup size, and GCN runs 64 only.
        if (Simulates("sub")) {
            vk13_props.requiredSubgroupSizeStages = vk::ShaderStageFlagBits::eCompute;
        }
        if (simulated == SimulatedGpu::Gcn && Simulates("sub")) {
            vk13_props.minSubgroupSize = 64;
        }
        if (simulated == SimulatedGpu::Gcn && Simulates("push")) {
            properties.limits.maxPushConstantsSize = 128;
        }
    }
    LOG_INFO(Render_Vulkan, "Physical device subgroup size {}", vk11_props.subgroupSize);

    if (available_extensions.empty()) {
        LOG_CRITICAL(Render_Vulkan, "No extensions supported by device.");
        return false;
    }

    boost::container::static_vector<const char*, 48> enabled_extensions;
    const auto add_extension = [&](std::string_view extension) -> bool {
        const auto result =
            std::find_if(available_extensions.begin(), available_extensions.end(),
                         [&](const std::string& name) { return name == extension; });
        // Already enabled (the DLSS plugin may ask for one the renderer uses).
        if (std::ranges::any_of(enabled_extensions,
                                [&](const char* name) { return extension == name; })) {
            return true;
        }

        if (result != available_extensions.end()) {
            LOG_INFO(Render_Vulkan, "Enabling extension: {}", extension);
            enabled_extensions.push_back(extension.data());
            return true;
        }

        LOG_WARNING(Render_Vulkan, "Extension {} unavailable.", extension);
        return false;
    };

    // Required
    ASSERT_MSG(add_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}", VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    ASSERT_MSG(add_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}", VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    ASSERT_MSG(add_extension(VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}",
               VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME);
    ASSERT_MSG(add_extension(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}", VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);

    const auto robustness2_features = feature_chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>();
    ASSERT_MSG(robustness2_features.robustBufferAccess2,
               "Required Vulkan feature unavailable: robustBufferAccess2");
    ASSERT_MSG(robustness2_features.robustImageAccess2,
               "Required Vulkan feature unavailable: robustImageAccess2");
    ASSERT_MSG(robustness2_features.nullDescriptor,
               "Required Vulkan feature unavailable: nullDescriptor");

    // Optional
    maintenance_5 = add_extension(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    maintenance_8 = add_extension(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
    // PERF-210 (RYGT_ANTI_LAG=1): AMD Anti-Lag, see AntiLagUpdate.
    if (const char* env = std::getenv("RYGT_ANTI_LAG"); env && env[0] == '1') {
        const auto chain = physical_device.getFeatures2<vk::PhysicalDeviceFeatures2,
                                                        vk::PhysicalDeviceAntiLagFeaturesAMD>();
        anti_lag = chain.get<vk::PhysicalDeviceAntiLagFeaturesAMD>().antiLag &&
                   add_extension(VK_AMD_ANTI_LAG_EXTENSION_NAME);
        LOG_WARNING(Render_Vulkan, "PERF-210: AMD Anti-Lag {}",
                    anti_lag ? "on" : "not supported by this driver");
    }
    attachment_feedback_loop = add_extension(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME);
    if (attachment_feedback_loop) {
        attachment_feedback_loop =
            add_extension(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
        if (!attachment_feedback_loop) {
            // We want both extensions so remove the first if the second isn't available
            enabled_extensions.pop_back();
        }
    }
    depth_range_unrestricted = add_extension(VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME);
    dynamic_state_3 = add_extension(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
    // PERF-017: build runtime pipelines from separately compiled stage libraries.
    if (feature_chain.get<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>()
            .graphicsPipelineLibrary &&
        Common::PerfFeatureEnabled(17) && add_extension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME)) {
        graphics_pipeline_library = add_extension(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
    }
    if (dynamic_state_3) {
        dynamic_state_3_features =
            feature_chain.get<vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT>();
        LOG_INFO(Render_Vulkan, "- extendedDynamicState3ColorWriteMask: {}",
                 dynamic_state_3_features.extendedDynamicState3ColorWriteMask);
    }
    custom_border_color = add_extension(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    depth_clip_control = add_extension(VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME);
    depth_clip_enable = add_extension(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
    vertex_input_dynamic_state = add_extension(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
    // Games excessively leave restart enabled with lists when it isn't actually needed, and it
    // slows down KosmicKrisp due to unrolling. Revisit if KosmicKrisp moves this to mesh shaders.
    list_restart = driver_id != vk::DriverId::eMesaKosmickrisp &&
                   add_extension(VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME);
    if (list_restart) {
        list_restart_features =
            feature_chain.get<vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT>();
        LOG_INFO(Render_Vulkan, "- primitiveTopologyListRestart: {}",
                 list_restart_features.primitiveTopologyListRestart);
        LOG_INFO(Render_Vulkan, "- primitiveTopologyPatchListRestart: {}",
                 list_restart_features.primitiveTopologyPatchListRestart);
    }
    // AMD-311: SHADGT_AMD_VERTEX_ORDER=0..5 picks the vertex order of AMD's per-vertex reads
    // (see spirv_emit_context.cpp); 0, the order an RX 7600 uses, unless set.
    if (const char* order_env = std::getenv("SHADGT_AMD_VERTEX_ORDER");
        order_env && order_env[0] >= '0' && order_env[0] <= '5') {
        amd_vertex_order = u32(order_env[0] - '0');
    }
    LOG_INFO(Render_Vulkan, "AMD-311: vertex order of AMD per-vertex reads: {}",
             amd_vertex_order);
    amd_shader_explicit_vertex_parameter =
        add_extension(VK_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_EXTENSION_NAME);
    const bool direct_barycentric_candidate =
        properties.vendorID == 0x1002 && SingleSampleBarycentricRequested();
    if (!amd_shader_explicit_vertex_parameter || direct_barycentric_candidate) {
        // Keep AMD explicit-vertex support for multisampled shaders; the opt-in candidate
        // additionally enables KHR for separately specialized one-sample fragments.
        fragment_shader_barycentric =
            add_extension(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
    }
    {
        // AMD-307: with AMD's barycentric extension and its older shader compiler (RX 6000 and
        // older), per-vertex reads find the driver's vertex order per triangle.
        // SHADGT_VERTEX_ORDER=0 turns it off, =1 turns it on with AMD's extension on any driver.
        const char* order_env = std::getenv("SHADGT_VERTEX_ORDER");
        vertex_order_check = amd_shader_explicit_vertex_parameter &&
                             (order_env ? order_env[0] == '1' : IsAmdLegacyShaderCompiler());
        LOG_INFO(Render_Vulkan, "AMD-307: barycentrics from {}, vertex order check {}",
                 amd_shader_explicit_vertex_parameter ? "VK_AMD_shader_explicit_vertex_parameter"
                 : fragment_shader_barycentric       ? "VK_KHR_fragment_shader_barycentric"
                                                     : "none",
                 vertex_order_check);
    }
    provoking_vertex = add_extension(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME);
    shader_stencil_export = add_extension(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME);
    image_load_store_lod = add_extension(VK_AMD_SHADER_IMAGE_LOAD_STORE_LOD_EXTENSION_NAME);
    amd_gcn_shader = add_extension(VK_AMD_GCN_SHADER_EXTENSION_NAME);
    amd_shader_trinary_minmax = add_extension(VK_AMD_SHADER_TRINARY_MINMAX_EXTENSION_NAME);
    // AMD-302: translated shaders that use 16-bit floats declare Float16, which needs
    // shaderFloat16 or AMD's half-float extension. RX 400/500 (Polaris) have only the extension.
    if (!feature_chain.get<vk::PhysicalDeviceVulkan12Features>().shaderFloat16) {
        add_extension(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME);
    }
    nv_framebuffer_mixed_samples = add_extension(VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME);
    amd_mixed_attachment_samples = add_extension(VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME);
    shader_atomic_float = add_extension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    shader_atomic_float2 = add_extension(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
    if (shader_atomic_float2) {
        shader_atomic_float2_features =
            feature_chain.get<vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>();
        // AMD-308: of the features the shader translator uses, the only one an RX 6000 (RDNA2)
        // has and an RX 7600 (RDNA3) lacks is float min/max on images; the format arrays of the
        // two are the same. With AMD's older shader compiler, image float min/max is made from
        // integer atomics as on the RX 7600, so both run the same translated shaders there.
        // SHADGT_FP32_IMAGE_ATOMICS=1 keeps the native atomics.
        if (const char* env = std::getenv("SHADGT_FP32_IMAGE_ATOMICS");
            env ? env[0] == '0' : IsAmdLegacyShaderCompiler()) {
            shader_atomic_float2_features.shaderImageFloat32AtomicMinMax = false;
            LOG_INFO(Render_Vulkan, "AMD-308: image float min/max from integer atomics");
        }
        LOG_INFO(Render_Vulkan, "- shaderBufferFloat32AtomicMinMax: {}",
                 shader_atomic_float2_features.shaderBufferFloat32AtomicMinMax);
        LOG_INFO(Render_Vulkan, "- shaderImageFloat32AtomicMinMax: {}",
                 shader_atomic_float2_features.shaderImageFloat32AtomicMinMax);
    }
    // P1 (prototype): guest memory imported into Vulkan, for the GPU to read in place.
    external_memory_host = add_extension(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    workgroup_memory_explicit_layout =
        add_extension(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
    if (workgroup_memory_explicit_layout) {
        workgroup_memory_explicit_layout_features =
            feature_chain.get<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
        LOG_INFO(Render_Vulkan, "- workgroupMemoryExplicitLayout: {}",
                 workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout);
        LOG_INFO(Render_Vulkan, "- workgroupMemoryExplicitLayoutScalarBlockLayout: {}",
                 workgroup_memory_explicit_layout_features
                     .workgroupMemoryExplicitLayoutScalarBlockLayout);
        LOG_INFO(
            Render_Vulkan, "- workgroupMemoryExplicitLayout16BitAccess: {}",
            workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout16BitAccess);
    }
    image_2d_view_of_3d = add_extension(VK_EXT_IMAGE_2D_VIEW_OF_3D_EXTENSION_NAME);
    if (image_2d_view_of_3d) {
        image_2d_view_of_3d_features =
            feature_chain.get<vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT>();
        LOG_INFO(Render_Vulkan, "- image2DViewOf3D: {}",
                 image_2d_view_of_3d_features.image2DViewOf3D);
        LOG_INFO(Render_Vulkan, "- sampler2DViewOf3D: {}",
                 image_2d_view_of_3d_features.sampler2DViewOf3D);
    }
    image_view_min_lod = add_extension(VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME);
    supports_memory_budget = add_extension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    shader_clock = add_extension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
    // DIAG-243: RYGT_DEVICE_FAULT=1 enables VK_EXT_device_fault, so a lost device reports the
    // faulting address and the driver's description.
    if (const char* env = std::getenv("RYGT_DEVICE_FAULT"); env && env[0] == '1') {
        const auto fault_chain =
            physical_device.getFeatures2<vk::PhysicalDeviceFeatures2,
                                         vk::PhysicalDeviceFaultFeaturesEXT>();
        if (fault_chain.get<vk::PhysicalDeviceFaultFeaturesEXT>().deviceFault) {
            device_fault = add_extension(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
        }
        LOG_WARNING(Render_Vulkan, "DIAG-243: VK_EXT_device_fault {}",
                    device_fault ? "enabled" : "not available");
    }
    if (shader_clock) {
        shader_clock_features = feature_chain.get<vk::PhysicalDeviceShaderClockFeaturesKHR>();
        LOG_INFO(Render_Vulkan, "- shaderSubgroupClock: {}",
                 shader_clock_features.shaderSubgroupClock);
    }
    // AMD-306: with AMD's older shader compiler (RX 6000 and older) pixel shaders keep whole
    // quads in branches (see emit_spirv.cpp). RX 7000 and newer compile with LLPC, which drew
    // these textures right without it, and an RX 7600's driver crashed compiling GT Sport's
    // garage shaders with it. SHADGT_QUAD_CONTROL=0 turns it off, =1 turns it on for any GPU.
    {
        const char* quad_env = std::getenv("SHADGT_QUAD_CONTROL");
        const bool wanted = quad_env ? quad_env[0] == '1' : IsAmdLegacyShaderCompiler();
        quad_control =
            wanted &&
            feature_chain.get<vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR>()
                .shaderMaximalReconvergence &&
            feature_chain.get<vk::PhysicalDeviceShaderQuadControlFeaturesKHR>().shaderQuadControl &&
            (GetSimulatedGpu() == SimulatedGpu::None ||
             !Simulates(VK_KHR_SHADER_QUAD_CONTROL_EXTENSION_NAME)) &&
            add_extension(VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME) &&
            add_extension(VK_KHR_SHADER_QUAD_CONTROL_EXTENSION_NAME);
        LOG_INFO(Render_Vulkan, "AMD-306: whole quads in pixel shader branches: {} (driver {})",
                 quad_control, driver_info);
    }
    // AMD-316: one line naming the Radeon generation, the shader compiler and the AMD fixes this
    // device runs with, so a log shows at once which path a card took.
    if (driver_id == vk::DriverId::eAmdProprietary) {
        const std::string model{GetModelName()};
        const auto has = [&](std::string_view text) {
            return model.find(text) != std::string::npos;
        };
        const char* generation = has("Radeon RX 9")   ? "RDNA4 (RX 9000)"
                                 : has("Radeon RX 7") ? "RDNA3 (RX 7000)"
                                 : has("Radeon RX 6") ? "RDNA2 (RX 6000)"
                                 : has("Radeon RX 55") || has("Radeon RX 56") || has("Radeon RX 57")
                                     ? "RDNA1 (RX 5000)"
                                 : has("Vega") ? "GCN 5 (Vega)"
                                 : has("Radeon RX 4") || has("Radeon RX 5") ? "GCN 4 (RX 400/500)"
                                                                            : "other Radeon";
        const bool legacy = IsAmdLegacyShaderCompiler();
        LOG_INFO(Render_Vulkan,
                 "AMD-316: {} on driver {} ({}), shader compiler {}; quad control {}, vertex "
                 "order check {}, KHR barycentrics {}",
                 generation, GetDriverVersionName(), driver_info, legacy ? "older (non-LLPC)" : "LLPC",
                 quad_control, vertex_order_check, fragment_shader_barycentric);
        if (legacy && !quad_control && !std::getenv("SHADGT_QUAD_CONTROL")) {
            LOG_WARNING(Render_Vulkan,
                        "AMD-316: this driver has no VK_KHR_shader_quad_control; textures in "
                        "shader branches may pick wrong mip levels. Updating the AMD driver may "
                        "help.");
        }
    }
    const bool calibrated_timestamps =
        TRACY_GPU_ENABLED ? add_extension(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME) : false;
#ifdef RYTY_GPU
    // ryGT: the device extensions NVIDIA DLSS needs, when the DLSS plugin is used.
    RytyDlss::AddDeviceExtensions(*instance, physical_device,
                                  [&](std::string_view name) { return add_extension(name); });
#endif

    const auto family_properties = physical_device.getQueueFamilyProperties();
    if (family_properties.empty()) {
        LOG_CRITICAL(Render_Vulkan, "Physical device reported no queues.");
        return false;
    }

    bool graphics_queue_found = false;
    for (std::size_t i = 0; i < family_properties.size(); i++) {
        const u32 index = static_cast<u32>(i);
        if (family_properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            queue_family_index = index;
            graphics_queue_found = true;
        }
    }

    if (!graphics_queue_found) {
        LOG_CRITICAL(Render_Vulkan, "Unable to find graphics and/or present queues.");
        return false;
    }

    // ryGT: with frame generation on, three more queues of the graphics family for its
    // swapchain's own threads (async compute, present, image acquire), when the family has them.
    static constexpr std::array queue_priorities = {1.0f, 1.0f, 1.0f, 1.0f};
    const u32 num_queues = EmulatorSettings.IsFrameGenEnabled() &&
                                   family_properties[queue_family_index].queueCount >= 4
                               ? 4u
                               : 1u;
    const vk::DeviceQueueCreateInfo queue_info = {
        .queueFamilyIndex = queue_family_index,
        .queueCount = num_queues,
        .pQueuePriorities = queue_priorities.data(),
    };

    const auto vk11_features = feature_chain.get<vk::PhysicalDeviceVulkan11Features>();
    vk12_features = feature_chain.get<vk::PhysicalDeviceVulkan12Features>();
    vk13_features = feature_chain.get<vk::PhysicalDeviceVulkan13Features>();
    vk::StructureChain device_chain = {
        vk::DeviceCreateInfo{
            .queueCreateInfoCount = 1u,
            .pQueueCreateInfos = &queue_info,
            .enabledExtensionCount = static_cast<u32>(enabled_extensions.size()),
            .ppEnabledExtensionNames = enabled_extensions.data(),
        },
        vk::PhysicalDeviceFeatures2{
            .features{
                .robustBufferAccess = features.robustBufferAccess,
                .imageCubeArray = features.imageCubeArray,
                .independentBlend = features.independentBlend,
                .geometryShader = features.geometryShader,
                .tessellationShader = features.tessellationShader,
                .sampleRateShading = features.sampleRateShading,
                .dualSrcBlend = features.dualSrcBlend,
                .logicOp = features.logicOp,
                .multiDrawIndirect = features.multiDrawIndirect,
                // NAT-161: GT7's indirect draws (their arguments written by compute shaders)
                // may start at an instance other than 0, which the shaders read as BaseInstance.
                .drawIndirectFirstInstance = features.drawIndirectFirstInstance,
                .depthClamp = features.depthClamp,
                .depthBiasClamp = features.depthBiasClamp,
                .fillModeNonSolid = features.fillModeNonSolid,
                .depthBounds = features.depthBounds,
                .wideLines = features.wideLines,
                .multiViewport = features.multiViewport,
                .samplerAnisotropy = features.samplerAnisotropy,
                .vertexPipelineStoresAndAtomics = features.vertexPipelineStoresAndAtomics,
                .fragmentStoresAndAtomics = features.fragmentStoresAndAtomics,
                .shaderImageGatherExtended = features.shaderImageGatherExtended,
                .shaderStorageImageExtendedFormats = features.shaderStorageImageExtendedFormats,
                .shaderStorageImageMultisample = features.shaderStorageImageMultisample,
                .shaderClipDistance = features.shaderClipDistance,
                .shaderFloat64 = features.shaderFloat64,
                .shaderInt64 = features.shaderInt64,
                .shaderInt16 = features.shaderInt16,
                .sparseBinding = features.sparseBinding,
                .sparseResidencyBuffer = features.sparseResidencyBuffer,
            },
        },
        vk::PhysicalDeviceVulkan11Features{
            .storageBuffer16BitAccess = vk11_features.storageBuffer16BitAccess,
            .uniformAndStorageBuffer16BitAccess = vk11_features.uniformAndStorageBuffer16BitAccess,
            .shaderDrawParameters = vk11_features.shaderDrawParameters,
        },
        vk::PhysicalDeviceVulkan12Features{
            .samplerMirrorClampToEdge = vk12_features.samplerMirrorClampToEdge,
            .drawIndirectCount = vk12_features.drawIndirectCount,
            .storageBuffer8BitAccess = vk12_features.storageBuffer8BitAccess,
            .uniformAndStorageBuffer8BitAccess = vk12_features.uniformAndStorageBuffer8BitAccess,
            .shaderBufferInt64Atomics = vk12_features.shaderBufferInt64Atomics,
            .shaderSharedInt64Atomics = vk12_features.shaderSharedInt64Atomics,
            .shaderFloat16 = vk12_features.shaderFloat16,
            .shaderInt8 = vk12_features.shaderInt8,
            .scalarBlockLayout = vk12_features.scalarBlockLayout,
            .uniformBufferStandardLayout = vk12_features.uniformBufferStandardLayout,
            .separateDepthStencilLayouts = vk12_features.separateDepthStencilLayouts,
            .hostQueryReset = vk12_features.hostQueryReset,
            .timelineSemaphore = vk12_features.timelineSemaphore,
            .bufferDeviceAddress = vk12_features.bufferDeviceAddress,
            .shaderOutputLayer = vk12_features.shaderOutputLayer,
        },
        vk::PhysicalDeviceVulkan13Features{
            .robustImageAccess = vk13_features.robustImageAccess,
            // FIX-188: Intel Arc (#65).
            .pipelineCreationCacheControl =
                IsIntel() && vk13_features.pipelineCreationCacheControl,
            .shaderDemoteToHelperInvocation = vk13_features.shaderDemoteToHelperInvocation,
            .subgroupSizeControl = vk13_features.subgroupSizeControl,
            .computeFullSubgroups = IsIntel() && vk13_features.computeFullSubgroups,
            .synchronization2 = vk13_features.synchronization2,
            .dynamicRendering = vk13_features.dynamicRendering,
            .maintenance4 = vk13_features.maintenance4,
        },
        // Extensions
        vk::PhysicalDeviceCustomBorderColorFeaturesEXT{
            .customBorderColors = true,
            .customBorderColorWithoutFormat = true,
        },
        vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT{
            .extendedDynamicState3ColorWriteMask =
                dynamic_state_3_features.extendedDynamicState3ColorWriteMask,
        },
        vk::PhysicalDeviceDepthClipControlFeaturesEXT{
            .depthClipControl = true,
        },
        vk::PhysicalDeviceDepthClipEnableFeaturesEXT{
            .depthClipEnable = true,
        },
        vk::PhysicalDeviceRobustness2FeaturesEXT{
            .robustBufferAccess2 = true,
            .robustImageAccess2 = true,
            .nullDescriptor = true,
        },
        vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT{
            .vertexInputDynamicState = true,
        },
        vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT{
            .primitiveTopologyListRestart = list_restart_features.primitiveTopologyListRestart,
            .primitiveTopologyPatchListRestart =
                list_restart_features.primitiveTopologyPatchListRestart,
        },
        vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR{
            .fragmentShaderBarycentric = true,
        },
        vk::PhysicalDeviceProvokingVertexFeaturesEXT{
            .provokingVertexLast = true,
        },
        vk::PhysicalDeviceVertexAttributeDivisorFeatures{
            .vertexAttributeInstanceRateDivisor = true,
        },
        vk::PhysicalDeviceMaintenance5FeaturesKHR{
            .maintenance5 = true,
        },
        vk::PhysicalDeviceMaintenance8FeaturesKHR{
            .maintenance8 = true,
        },
        vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT{
            .attachmentFeedbackLoopLayout = true,
        },
        vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT{
            .attachmentFeedbackLoopDynamicState = true,
        },
        vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT{
            .shaderBufferFloat32AtomicMinMax =
                shader_atomic_float2_features.shaderBufferFloat32AtomicMinMax,
            .shaderImageFloat32AtomicMinMax =
                shader_atomic_float2_features.shaderImageFloat32AtomicMinMax,
        },
        vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR{
            .workgroupMemoryExplicitLayout =
                workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout,
            .workgroupMemoryExplicitLayoutScalarBlockLayout =
                workgroup_memory_explicit_layout_features
                    .workgroupMemoryExplicitLayoutScalarBlockLayout,
            .workgroupMemoryExplicitLayout16BitAccess =
                workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout16BitAccess,
        },
        vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT{
            .image2DViewOf3D = image_2d_view_of_3d_features.image2DViewOf3D,
            .sampler2DViewOf3D = image_2d_view_of_3d_features.sampler2DViewOf3D,
        },
        vk::PhysicalDeviceImageViewMinLodFeaturesEXT{
            .minLod = true,
        },
        vk::PhysicalDeviceShaderClockFeaturesKHR{
            .shaderSubgroupClock = shader_clock_features.shaderSubgroupClock,
        },
        vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT{
            .graphicsPipelineLibrary = true,
        },
        vk::PhysicalDeviceAntiLagFeaturesAMD{
            .antiLag = true, // PERF-210
        },
        vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR{
            .shaderMaximalReconvergence = true, // AMD-306
        },
        vk::PhysicalDeviceShaderQuadControlFeaturesKHR{
            .shaderQuadControl = true, // AMD-306
        },
    };
    if (!quad_control) {
        device_chain.unlink<vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR>();
        device_chain.unlink<vk::PhysicalDeviceShaderQuadControlFeaturesKHR>();
    }
    if (!anti_lag) {
        device_chain.unlink<vk::PhysicalDeviceAntiLagFeaturesAMD>();
    }
    if (!graphics_pipeline_library) {
        device_chain.unlink<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>();
    }

    if (!custom_border_color) {
        device_chain.unlink<vk::PhysicalDeviceCustomBorderColorFeaturesEXT>();
    }
    if (!dynamic_state_3) {
        device_chain.unlink<vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT>();
    }
    if (!depth_clip_control) {
        device_chain.unlink<vk::PhysicalDeviceDepthClipControlFeaturesEXT>();
    }
    if (!depth_clip_enable) {
        device_chain.unlink<vk::PhysicalDeviceDepthClipEnableFeaturesEXT>();
    }
    if (!vertex_input_dynamic_state) {
        device_chain.unlink<vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT>();
    }
    if (!list_restart) {
        device_chain.unlink<vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT>();
    }
    if (!fragment_shader_barycentric) {
        device_chain.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
    }
    if (!provoking_vertex) {
        device_chain.unlink<vk::PhysicalDeviceProvokingVertexFeaturesEXT>();
    }
    if (!maintenance_5) {
        device_chain.unlink<vk::PhysicalDeviceMaintenance5FeaturesKHR>();
    }
    if (!maintenance_8) {
        device_chain.unlink<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
    }
    if (!attachment_feedback_loop) {
        device_chain.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>();
        device_chain.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>();
    }
    if (!shader_atomic_float2) {
        device_chain.unlink<vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>();
    }
    if (!workgroup_memory_explicit_layout) {
        device_chain.unlink<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
    }
    if (!image_2d_view_of_3d) {
        device_chain.unlink<vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT>();
    }
    if (!image_view_min_lod) {
        device_chain.unlink<vk::PhysicalDeviceImageViewMinLodFeaturesEXT>();
    }
    if (!shader_clock) {
        device_chain.unlink<vk::PhysicalDeviceShaderClockFeaturesKHR>();
    }

    // DIAG-243: the fault feature joins the chain when its extension is on.
    vk::PhysicalDeviceFaultFeaturesEXT fault_features{.deviceFault = true};
    if (device_fault) {
        auto& create_info = device_chain.get<vk::DeviceCreateInfo>();
        fault_features.pNext = const_cast<void*>(create_info.pNext);
        create_info.pNext = &fault_features;
    }
    auto [device_result, dev] = physical_device.createDeviceUnique(device_chain.get());
    if (device_result != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "Failed to create device: {}", vk::to_string(device_result));
        return false;
    }
    device = std::move(dev);
    // AMD-300: with "driver" skipped, the driver keeps both extensions on while the renderer
    // takes the paths a card without them takes (renderer path or driver mode?).
    if (GetSimulatedGpu() == SimulatedGpu::Gcn && !Simulates("driver")) {
        graphics_pipeline_library = graphics_pipeline_library && !Simulates(
                                        VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
        vertex_input_dynamic_state = vertex_input_dynamic_state && !Simulates(
                                         VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

    graphics_queue = device->getQueue(queue_family_index, 0);
    present_queue = device->getQueue(queue_family_index, 0);
    framegen_queues.clear();
    for (u32 i = 1; i < num_queues; ++i) {
        framegen_queues.push_back(device->getQueue(queue_family_index, i));
    }

    if (calibrated_timestamps) {
        const auto [time_domains_result, time_domains] =
            physical_device.getCalibrateableTimeDomainsEXT();
        if (time_domains_result == vk::Result::eSuccess) {
#if _WIN64
            const bool has_host_time_domain =
                std::find(time_domains.cbegin(), time_domains.cend(),
                          vk::TimeDomainEXT::eQueryPerformanceCounter) != time_domains.cend();
#elif __linux__
            const bool has_host_time_domain =
                std::find(time_domains.cbegin(), time_domains.cend(),
                          vk::TimeDomainEXT::eClockMonotonicRaw) != time_domains.cend();
#else
            // Tracy limitation means only Windows and Linux can use host time domain.
            // https://github.com/shadps4-emu/tracy/blob/c6d779d78508514102fbe1b8eb28bda10d95bb2a/public/tracy/TracyVulkan.hpp#L384-L389
            const bool has_host_time_domain = false;
#endif
            if (has_host_time_domain) {
                static constexpr std::string_view context_name{"vk_rasterizer"};
                profiler_context = TracyVkContextHostCalibrated(
                    *instance, physical_device, *device,
                    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
                    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr);
                TracyVkContextName(profiler_context, context_name.data(), context_name.size());
            }
        } else {
            LOG_WARNING(Render_Vulkan, "Could not query calibrated time domains for profiling: {}",
                        vk::to_string(time_domains_result));
        }
    }

    CreateAllocator();
    return true;
}

void Instance::CreateAllocator() {
    const VmaVulkanFunctions functions = {
        .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
    };

    const VmaAllocatorCreateInfo allocator_info = {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = physical_device,
        .device = *device,
        .pVulkanFunctions = &functions,
        .instance = *instance,
        .vulkanApiVersion = TargetVulkanApiVersion,
    };

    const VkResult result = vmaCreateAllocator(&allocator_info, &allocator);
    if (result != VK_SUCCESS) {
        UNREACHABLE_MSG("Failed to initialize VMA with error {}",
                        vk::to_string(vk::Result{result}));
    }
}

void Instance::CollectDeviceParameters() {
    const vk::StructureChain property_chain =
        physical_device
            .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDriverProperties>();
    const vk::PhysicalDeviceDriverProperties driver =
        property_chain.get<vk::PhysicalDeviceDriverProperties>();

    driver_id = driver.driverID;
    vendor_name = driver.driverName.data();
    driver_info = driver.driverInfo.data(); // AMD-306/307

    const std::string model_name{GetModelName()};
    const std::string driver_version = GetDriverVersionName();
    const std::string driver_name = fmt::format("{} {}", vendor_name, driver_version);
    const std::string api_version = GetReadableVersion(properties.apiVersion);
    const std::string extensions = fmt::format("{}", fmt::join(available_extensions, ", "));

    LOG_INFO(Render_Vulkan, "GPU_Vendor: {}", vendor_name);
    LOG_INFO(Render_Vulkan, "GPU_Model: {}", model_name);
    LOG_INFO(Render_Vulkan, "GPU_Integrated: {}", IsIntegrated() ? "Yes" : "No");
    LOG_INFO(Render_Vulkan, "GPU_Vulkan_Driver: {}", driver_name);
    LOG_INFO(Render_Vulkan, "GPU_Vulkan_Version: {}", api_version);
    LOG_INFO(Render_Vulkan, "GPU_Vulkan_Extensions: {}", extensions);
}

void Instance::CollectPhysicalMemoryInfo() {
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    vk::PhysicalDeviceMemoryProperties2 props = {
        .pNext = supports_memory_budget ? &budget : nullptr,
    };
    physical_device.getMemoryProperties2(&props);
    const auto& memory_props = props.memoryProperties;
    const size_t num_props = memory_props.memoryHeapCount;
    total_memory_budget = 0;
    u64 device_initial_usage = 0;
    u64 local_memory = 0;
    for (size_t i = 0; i < num_props; ++i) {
        const bool is_device_local =
            (memory_props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) !=
            vk::MemoryHeapFlags{};
        if (!IsIntegrated() && !is_device_local) {
            // Ignore non-device local memory on discrete GPUs.
            continue;
        }
        valid_heaps.push_back(i);
        if (is_device_local) {
            local_memory += memory_props.memoryHeaps[i].size;
        }
        if (supports_memory_budget) {
            device_initial_usage += budget.heapUsage[i];
            total_memory_budget += budget.heapBudget[i];
            continue;
        }
        // If memory budget is not supported, use the size of the heap as the budget.
        total_memory_budget += memory_props.memoryHeaps[i].size;
    }
    if (!IsIntegrated()) {
        // We reserve some memory for the system.
        const u64 system_memory = std::min<u64>(total_memory_budget / 8, 1_GB);
        total_memory_budget -= system_memory;
        return;
    }
    // Leave at least 8 GB for the system on integrated GPUs.
    const s64 available_memory = static_cast<s64>(total_memory_budget - device_initial_usage);
    total_memory_budget =
        static_cast<u64>(std::max<s64>(available_memory - 8_GB, static_cast<s64>(2_GB)));
}

// NAT-VRAM-001: the launcher's video memory setting (GPU.vram_budget_mb, 0 = Auto) or
// RYGT_VRAM_BUDGET_MB caps the budget the texture cache's garbage collector plans with, so it
// frees cached images the game hasn't used lately once the process uses more than the setting.
// Auto keeps the driver-reported budget (VK_EXT_memory_budget) unchanged.
u64 Instance::VideoMemorySettingBytes() {
    static const u64 bytes = [] {
        if (const char* env = std::getenv("RYGT_VRAM_BUDGET_MB"); env && *env) {
            return u64{std::strtoull(env, nullptr, 10)} << 20;
        }
        if (const auto value = Common::RytyGameConfigValue("vram_budget_mb")) {
            return u64{std::strtoull(value->c_str(), nullptr, 10)} << 20;
        }
        return u64{0};
    }();
    return bytes;
}

void Instance::ApplyVideoMemorySetting() {
    const u64 driver_budget = total_memory_budget;
    const u64 setting = VideoMemorySettingBytes();
    if (setting != 0) {
        // Not below 1 GB: the renderer's own targets and buffers need about that much.
        total_memory_budget = std::min(driver_budget, std::max<u64>(setting, 1_GB));
    }
    LOG_WARNING(Render_Vulkan,
                "NAT-VRAM-001: video memory budget {} MB ({}; driver budget {} MB, memory budget "
                "extension {})",
                total_memory_budget >> 20,
                setting != 0 ? fmt::format("setting {} MB", setting >> 20) : std::string{"Auto"},
                driver_budget >> 20, supports_memory_budget ? "on" : "off");
}

void Instance::CollectImageFormatInfo() {
    // Check for block texel view support using basic image format info.
    const vk::PhysicalDeviceImageFormatInfo2 block_texel_view_info{
        .format = vk::Format::eBc1RgbaUnormBlock,
        .type = vk::ImageType::e2D,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled,
        .flags = vk::ImageCreateFlagBits::eBlockTexelViewCompatible,
    };
    const auto block_texel_view_props =
        physical_device.getImageFormatProperties2(block_texel_view_info);
    supports_block_texel_view = block_texel_view_props.result == vk::Result::eSuccess;
    LOG_INFO(Render_Vulkan, "Block Texel View support: {}",
             supports_block_texel_view ? "Yes" : "No");

    // Check and log format support details.
    for (const auto& format : LiverpoolToVK::SurfaceFormats()) {
        if (!IsFormatSupported(format.vk_format, format.flags)) {
            LOG_WARNING(Render_Vulkan,
                        "Surface format data_format={}, number_format={} is not fully supported "
                        "(vk_format={}, missing features={})",
                        static_cast<u32>(format.data_format),
                        static_cast<u32>(format.number_format), vk::to_string(format.vk_format),
                        vk::to_string(format.flags & ~GetFormatFeatureFlags(format.vk_format)));
        }
    }
    for (const auto& format : LiverpoolToVK::DepthFormats()) {
        if (!IsFormatSupported(format.vk_format, format.flags)) {
            LOG_WARNING(Render_Vulkan,
                        "Depth format z_format={}, stencil_format={} is not fully supported "
                        "(vk_format={}, missing features={})",
                        static_cast<u32>(format.z_format), static_cast<u32>(format.stencil_format),
                        vk::to_string(format.vk_format),
                        vk::to_string(format.flags & ~GetFormatFeatureFlags(format.vk_format)));
        }
    }
}

void Instance::CollectToolingInfo() const {
    if (driver_id == vk::DriverId::eAmdProprietary ||
        driver_id == vk::DriverId::eIntelProprietaryWindows) {
        // AMD: Causes issues with Reshade.
        // Intel: Causes crash on start.
        return;
    }
    const auto [tools_result, tools] = physical_device.getToolProperties();
    if (tools_result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Could not get Vulkan tool properties: {}",
                  vk::to_string(tools_result));
        return;
    }
    for (const vk::PhysicalDeviceToolProperties& tool : tools) {
        const std::string_view name = tool.name;
        LOG_INFO(Render_Vulkan, "Attached debugging tool: {}", name);
    }
}

u64 Instance::GetDeviceMemoryUsage() const {
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT memory_budget_props{};
    vk::PhysicalDeviceMemoryProperties2 props = {
        .pNext = &memory_budget_props,
    };
    physical_device.getMemoryProperties2(&props);

    u64 total_usage = 0;
    for (const size_t heap : valid_heaps) {
        total_usage += memory_budget_props.heapUsage[heap];
    }
    return total_usage;
}

vk::FormatFeatureFlags2 Instance::GetFormatFeatureFlags(vk::Format format) const {
    const auto it = format_properties.find(format);
    if (it == format_properties.end()) {
        UNIMPLEMENTED_MSG("Properties of format {} have not been queried.", vk::to_string(format));
    }

    return it->second.optimalTilingFeatures | it->second.bufferFeatures;
}

bool Instance::IsFormatSupported(const vk::Format format,
                                 const vk::FormatFeatureFlags2 flags) const {
    if (format == vk::Format::eUndefined) [[unlikely]] {
        return true;
    }
    return (GetFormatFeatureFlags(format) & flags) == flags;
}

vk::Format Instance::GetSupportedFormat(const vk::Format format,
                                        const vk::FormatFeatureFlags2 flags) const {
    if (!IsFormatSupported(format, flags)) [[unlikely]] {
        switch (format) {
        case vk::Format::eD16UnormS8Uint:
            if (IsFormatSupported(vk::Format::eD24UnormS8Uint, flags)) {
                return vk::Format::eD24UnormS8Uint;
            }
            if (IsFormatSupported(vk::Format::eD32SfloatS8Uint, flags)) {
                return vk::Format::eD32SfloatS8Uint;
            }
            break;
        case vk::Format::eR8Srgb:
            if (IsFormatSupported(vk::Format::eR8Unorm, flags)) {
                return vk::Format::eR8Unorm;
            }
            break;
        case vk::Format::eR8G8Srgb:
            if (IsFormatSupported(vk::Format::eR8G8Unorm, flags)) {
                return vk::Format::eR8G8Unorm;
            }
            break;
        default:
            break;
        }
    }
    return format;
}

namespace {
// DIAG-243: a ring of the last draws recorded (pipeline key hash, info word).
struct FaultDraw {
    std::atomic<u64> hash{};
    std::atomic<u32> info{};
    std::atomic<u64> serial{};
};
constexpr u32 FaultRingSize = 64;
std::array<FaultDraw, FaultRingSize> g_fault_draws;
std::atomic<u64> g_fault_serial{};
} // namespace

void NoteDrawForFault(u64 pipeline_hash, u32 info) {
    const u64 serial = g_fault_serial.fetch_add(1, std::memory_order_relaxed);
    auto& entry = g_fault_draws[serial % FaultRingSize];
    entry.hash.store(pipeline_hash, std::memory_order_relaxed);
    entry.info.store(info, std::memory_order_relaxed);
    entry.serial.store(serial + 1, std::memory_order_release);
}

void Instance::ReportDeviceLost() const {
    static std::atomic<bool> reported{};
    if (reported.exchange(true)) {
        return;
    }
    LOG_CRITICAL(Render_Vulkan, "DIAG-243: the GPU device was lost");
    if (device_fault && VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT) {
        const VkDevice dev = static_cast<VkDevice>(*device);
        VkDeviceFaultCountsEXT counts{.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT(dev, &counts, nullptr);
        std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
        std::vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
        VkDeviceFaultInfoEXT info{.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
        info.pAddressInfos = addresses.data();
        info.pVendorInfos = vendors.data();
        counts.vendorBinarySize = 0; // no vendor binary dump
        const VkResult result =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT(dev, &counts, &info);
        LOG_CRITICAL(Render_Vulkan, "DIAG-243: device fault ({}): {}; {} addresses, {} vendor infos",
                     static_cast<int>(result), info.description, counts.addressInfoCount,
                     counts.vendorInfoCount);
        // VkDeviceFaultAddressTypeEXT.
        static constexpr const char* types[] = {"none", "read invalid", "write invalid",
                                                "execute invalid",
                                                "instruction pointer (a running wave)",
                                                "instruction pointer invalid",
                                                "instruction pointer fault"};
        for (u32 i = 0; i < counts.addressInfoCount; ++i) {
            const auto& a = addresses[i];
            const u32 type = static_cast<u32>(a.addressType);
            LOG_CRITICAL(Render_Vulkan, "DIAG-243:   address {:#x} (+-{:#x}): {}",
                         a.reportedAddress, a.addressPrecision,
                         type < std::size(types) ? types[type] : "?");
        }
        for (u32 i = 0; i < counts.vendorInfoCount; ++i) {
            const auto& v = vendors[i];
            LOG_CRITICAL(Render_Vulkan, "DIAG-243:   vendor: {} (code {:#x}, data {:#x})",
                         v.description, v.vendorFaultCode, v.vendorFaultData);
        }
    }
    // The draws recorded last, newest first (the GPU was somewhere among them or before).
    const u64 newest = g_fault_serial.load(std::memory_order_acquire);
    for (u32 back = 0; back < FaultRingSize && back < newest; ++back) {
        const u64 serial = newest - 1 - back;
        const auto& entry = g_fault_draws[serial % FaultRingSize];
        if (entry.serial.load(std::memory_order_acquire) != serial + 1) {
            continue;
        }
        const u32 info = entry.info.load(std::memory_order_relaxed);
        LOG_CRITICAL(Render_Vulkan,
                     "DIAG-243:   draw -{}: pipeline {:#018x}, tessellated {}, LDS emulation {}",
                     back, entry.hash.load(std::memory_order_relaxed), (info >> 8) & 1,
                     info & 0xff);
    }
}

// PERF-210: the input stage runs on the game thread that submits the flip, so the driver can hold
// the game there and start its next frame later; the present stage closes that frame.
void Instance::AntiLagUpdate(bool present) const {
    if (!anti_lag) {
        return;
    }
    std::scoped_lock lock{anti_lag_mutex};
    // A game that flips only at end of pipe has no input stage: no frame to close either.
    if (present && !anti_lag_input_sent) {
        return;
    }
    anti_lag_input_sent = !present;
    if (!present) {
        ++anti_lag_frame;
    }
    const vk::AntiLagPresentationInfoAMD info{
        .stage = present ? vk::AntiLagStageAMD::ePresent : vk::AntiLagStageAMD::eInput,
        .frameIndex = anti_lag_frame,
    };
    device->antiLagUpdateAMD(vk::AntiLagDataAMD{
        .mode = vk::AntiLagModeAMD::eOn,
        .maxFPS = 0,
        .pPresentationInfo = &info,
    });
}

} // namespace Vulkan
