<!-- SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# shadGT7

The experimental Vulkan renderer used by [ryGT7](https://gitlab.com/rygt-team/rygt7) for
Gran Turismo 7 on Windows. Derived from [shadGT](https://github.com/mlgprorektm8/shadGT),
[shadPS4](https://github.com/shadps4-emu/shadPS4), and their contributors. Upstream Git history
and license notices are preserved.

Players should use the complete [ryGT7 prerelease](https://gitlab.com/rygt-team/rygt7/-/releases)
and its launcher. This repository supplies the renderer library, not a standalone GT7 setup.
You provide your own decrypted CUSA24769 game, firmware modules, and fonts.

The GT7 branch adds native rendering and compatibility fixes, including wave64 translation
that prevents overlapping crowd instance writes, and sparse texture uploads that preserve
resident streamed mip levels. These fixes were verified at Deep Forest. A fix for flickering night shadows at Yas Marina
(cached render targets reused after a memory remap) is included but not yet confirmed in game. Other tracks, updates,
and hardware may still have rendering errors or crashes.

## GT7 ultrawide, HUD editor, frame generation and fast flips

These come from dessierror's [wheel + ultrawide edition](https://gitlab.com/dessierror/rygt-7-online-build)
(its `renderer` branch, cherry-picked). Each is switchable; the switches are environment variables,
which the ryGT7 launcher sets from its options.

- Ultrawide (GT7-UW): with `GPU.aspect_ratio` wider than 16:9 in the game's custom config (or
  `RYGT_ASPECT`), the scene is stretched over the window and the HUD is kept at 16:9 in the centre.
  A frame's draws into a display buffer after its first (the scene) go to two HUD pictures, one over
  black and one over white, and the post-processing pass lays them over the scene with the coverage
  the game blended them with. `RYGT_HUD_SPLIT=0` turns the split off; `RYGT_HUD_STREAM=0` records
  the draws over white in the main list, switching attachments per draw, instead of in a list of
  their own.
- HUD editor: F7 in a race (ultrawide only) shows the HUD pieces as boxes to drag and resize; Save
  writes `user\custom_configs\hud_layout.ini`. Placed pieces apply to race HUDs only (told apart by
  their multisampled track and course maps), not to menus.
- Prepared draws for GT7 (GT7-PREP): hooks on GT7's (CUSA24769 1.00) PGL job begin, end and fork,
  installed only with `RYGT_NATIVE_PREPARE=1` or `RYGT_NATIVE_SHADOW=1` and only when the
  executable's bytes match. Ended jobs are walked on `RYGT_NATIVE_PREPARE_WORKERS` renderer threads
  (default 4; 0 walks on the job's thread, as before).
- DLSS multi-frame generation (GT-MFG): `RYGT_DLSSG_FRAMES=1..5` generated frames a game frame
  (default 1, as before; more than 1 needs an RTX 50 and falls back to 1 when refused).
- Fast flips (GT-FF): `RYGT_FAST_FLIPS=1` shows a frame as soon as it is submitted, no faster than
  the flip rate, instead of at the next vblank. Off by default.

## Build

Clone with submodules. In a Visual Studio x64 developer environment with clang-cl, CMake,
Ninja, Python and the Windows SDK:

```
cmake --preset x64-Clang-Release -DSHADGT_RYTY_GPU=ON -DHAVE_ENDIAN_H=OFF
cmake --build Build/x64-Clang-Release --target ryty_gpu --parallel 8
```

This builds libSceVideoOut.prx. The companion runtime and packaging instructions are in
[ryGT7 build-support](https://gitlab.com/rygt-team/rygt7/-/tree/main/build-support).
The standalone shadGT emulator and historical GT Sport documentation remain inherited source;
use the ryGT7 launcher to prepare and run GT7.

The wave64 guard regression test contains 512 synthetic checks and no dumped game shaders.
With tests enabled, its CMake target is shadgt_wave64_guard_regression.

## License and credits

GPL version 2 or later applies as stated in the source files; dependencies retain their own
licenses. See LICENSE and upstream contributor notices. Gran Turismo is a trademark of Sony
Interactive Entertainment. This project is independent of Sony and Polyphony Digital.
