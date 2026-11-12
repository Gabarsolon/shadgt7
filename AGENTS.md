# Notes for agents working on shadgt7

## RYGT_* and SHADGT_* switches need a description line

The ryGT7 launcher's Advanced page (and the RyGTS one, which reads shadGT the same way) lists the renderer's experimental switches by reading
`libSceVideoOut.prx` as a file: every `"RYGT_..."` or `"SHADGT_..."` string the renderer passes to `getenv`
(or `Common::DiagEnv`) becomes a row. The hover tooltip, editor and heading come from metadata
strings in the same DLL, kept in **`src/video_core/rygt_switches.cpp`**.

**When you add a switch** (a new `getenv("RYGT_...")` or `getenv("SHADGT_...")`, in `src/` or `ryty_gpu/`), add one line to
`RyGT_SwitchDescriptions` in that file, under the matching group comment:

```cpp
"RYGTSW1|RYGT_EXAMPLE_SWITCH|flag|Rendering workarounds|off|1 = on, 0 = off|One sentence on what it does.",
```

Fields, separated by `|` (only the description may contain `|`):

| Field | What to write |
|---|---|
| `RYGTSW1` | Format tag; always this. |
| name | The exact variable name. |
| kind | `flag` (0/1 or presence), `int`, `hex`, `float` or `string`. Picks the launcher's editor and validation. |
| group | One of: `Shader objects & caches`, `Shader compiling`, `Native recorder & sync`, `Memory & VRAM`, `Tessellation`, `Frame pacing & present`, `Rendering workarounds`, `AMD`, `NVIDIA`, `Runtime`, `Diagnostics & tracing` (hidden until the user asks; use it for logs, traces, dumps and stats). |
| default | What happens when the variable is not set (`off`, `on`, `300`, ...). |
| values | What the accepted values mean; say "Diagnostics builds only", "Debug builds only" or "any value turns it on" when true. |
| description | One plain sentence, about 150 characters at most, for a user deciding whether to set it. Say if it slows the game or writes large logs. |

**When you change or remove a switch**, update or delete its line.

- A switch without a line still shows in the launcher, under "Not described in this renderer build".
- A line whose name the build never reads is not shown.
- Keep the table `extern "C"` and exported (`RYGT_SWITCH_EXPORT`): the renderer never reads it, so
  the export is what stops the linker from dropping the strings.
- Keep the switch name a plain string literal at the `getenv` call. The launcher finds names that way,
  so a name built at run time (concatenated or formatted) will not be found.

The launcher side is `loader/AdvancedPage.cs` (`SwitchWalker`) and `loader/ADVANCED-PAGE.md` in ryGT7.
