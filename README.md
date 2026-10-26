<!-- SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# shadGT7

The experimental Vulkan renderer used by [ryGT7](https://gitlab.com/67933lance/rygt7) for
Gran Turismo 7 on Windows. Derived from [shadGT](https://github.com/mlgprorektm8/shadGT),
[shadPS4](https://github.com/shadps4-emu/shadPS4), and their contributors. Upstream Git history
and license notices are preserved.

Players should use the complete [ryGT7 prerelease](https://gitlab.com/67933lance/rygt7/-/releases)
and its launcher. This repository supplies the renderer library, not a standalone GT7 setup.
You provide your own decrypted CUSA24769 game, firmware modules, and fonts.

The GT7 branch adds native rendering and compatibility fixes, including wave64 translation
that prevents overlapping crowd instance writes, and sparse texture uploads that preserve
resident streamed mip levels. These fixes were verified at Deep Forest. A fix for flickering night shadows at Yas Marina
(cached render targets reused after a memory remap) is included but not yet confirmed in game. Other tracks, updates,
and hardware may still have rendering errors or crashes.

## Build

Clone with submodules. In a Visual Studio x64 developer environment with clang-cl, CMake,
Ninja, Python and the Windows SDK:

```
cmake --preset x64-Clang-Release -DSHADGT_RYTY_GPU=ON -DHAVE_ENDIAN_H=OFF
cmake --build Build/x64-Clang-Release --target ryty_gpu --parallel 8
```

This builds libSceVideoOut.prx. The companion runtime and packaging instructions are in
[ryGT7 build-support](https://gitlab.com/67933lance/rygt7/-/tree/main/build-support).
The standalone shadGT emulator and historical GT Sport documentation remain inherited source;
use the ryGT7 launcher to prepare and run GT7.

The wave64 guard regression test contains 512 synthetic checks and no dumped game shaders.
With tests enabled, its CMake target is shadgt_wave64_guard_regression.

## License and credits

GPL version 2 or later applies as stated in the source files; dependencies retain their own
licenses. See LICENSE and upstream contributor notices. Gran Turismo is a trademark of Sony
Interactive Entertainment. This project is independent of Sony and Polyphony Digital.
