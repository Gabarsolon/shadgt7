// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "common/hdr_settings.h"

TEST(HdrSettings, DefaultsDoNotAdjustPqOrSendMetadata) {
    for (const char* text : {"", "{", "{}", R"({"GPU":{}})", R"({"GPU":[]})"}) {
        const auto settings = Common::Hdr::ParseOutputSettings(text);
        EXPECT_FLOAT_EQ(settings.gain, 1.0f);
        EXPECT_FLOAT_EQ(settings.peak_nits, 0.0f);
        EXPECT_FLOAT_EQ(settings.metadata_nits, 0.0f);
    }
}

TEST(HdrSettings, OnlyRootGpuValuesApply) {
    const auto settings = Common::Hdr::ParseOutputSettings(
        R"({"hdr_gain":8,"other":{"hdr_peak_nits":100},"GPU":{"nested":{"hdr_gain":7}}})");
    EXPECT_FLOAT_EQ(settings.gain, 1.0f);
    EXPECT_FLOAT_EQ(settings.peak_nits, 0.0f);
}

TEST(HdrSettings, FiniteNumbersAndNumericStringsAreAccepted) {
    const auto settings = Common::Hdr::ParseOutputSettings(
        R"({"GPU":{"hdr_gain":" 2.5 ","hdr_peak_nits":900,"hdr_metadata_nits":1000,"hdr_metadata_fall_nits":400}})");
    EXPECT_FLOAT_EQ(settings.gain, 2.5f);
    EXPECT_FLOAT_EQ(settings.peak_nits, 900.0f);
    EXPECT_FLOAT_EQ(settings.metadata_nits, 1000.0f);
    EXPECT_FLOAT_EQ(settings.metadata_fall_nits, 400.0f);
}

TEST(HdrSettings, InvalidAndNonfiniteValuesFallBack) {
    for (const char* text : {
             R"({"GPU":{"hdr_gain":"nan","hdr_peak_nits":"inf","hdr_metadata_nits":null}})",
             R"({"GPU":{"hdr_gain":true,"hdr_peak_nits":{},"hdr_metadata_nits":[]}})",
             R"({"GPU":{"hdr_gain":"3x","hdr_peak_nits":"1e9999","hdr_metadata_nits":""}})"}) {
        const auto settings = Common::Hdr::ParseOutputSettings(text);
        EXPECT_FLOAT_EQ(settings.gain, 1.0f);
        EXPECT_FLOAT_EQ(settings.peak_nits, 0.0f);
        EXPECT_FLOAT_EQ(settings.metadata_nits, 0.0f);
    }
}

TEST(HdrSettings, LimitsKeepMetadataAverageBelowPeak) {
    const auto low = Common::Hdr::ParseOutputSettings(
        R"({"GPU":{"hdr_gain":-1,"hdr_peak_nits":-1,"hdr_metadata_nits":20,"hdr_metadata_fall_nits":500}})");
    EXPECT_FLOAT_EQ(low.gain, 0.1f);
    EXPECT_FLOAT_EQ(low.peak_nits, 0.0f);
    EXPECT_FLOAT_EQ(low.metadata_nits, 100.0f);
    EXPECT_FLOAT_EQ(low.metadata_fall_nits, 100.0f);
    const auto high = Common::Hdr::ParseOutputSettings(
        R"({"GPU":{"hdr_gain":99,"hdr_peak_nits":20000,"hdr_metadata_nits":20000,"hdr_metadata_fall_nits":-10}})");
    EXPECT_FLOAT_EQ(high.gain, 8.0f);
    EXPECT_FLOAT_EQ(high.peak_nits, 10000.0f);
    EXPECT_FLOAT_EQ(high.metadata_nits, 10000.0f);
    EXPECT_FLOAT_EQ(high.metadata_fall_nits, 1.0f);
}
