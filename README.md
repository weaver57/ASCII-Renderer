# ASCII Renderer

A high-performance RGB ASCII **video and image renderer** written in Rust. It
decodes a video or image, downsamples each frame, and streams colored ASCII art
to your terminal in real time using ANSI truecolor escape codes — all through a
single buffered write per frame.

```
Usage: ascii_renderer [OPTIONS] <FILE>
```

---

## Features

- **Static image viewer** with live interactive controls (color mode, ramp, invert).
- **Real-time video playback** (via FFmpeg) with pause/resume and loop.
- **Perceptual luminance** (Rec. 709 coefficients) for true-to-eye brightness mapping.
- **Aspect-ratio-correct grid sizing** — output is not vertically stretched or squashed.
- **Area-aware downsampling** (`Triangle` filter) to avoid aliasing on fine detail.
- **Edge-aware glyphs** — structural edges detected via full-resolution Sobel + adaptive Canny (NMS, percentile thresholds, queue hysteresis) render as correctly-oriented directional characters (`|`, `-`, `/`, `\`), falling back to brightness shading everywhere else.
- **Buffered output** — each frame is composed into one buffer and flushed in a single write.
- Three color modes and multiple adjustable ASCII ramps, switchable mid-run.

---

## Requirements

| Tool      | Purpose                                          | Required for                  |
|-----------|--------------------------------------------------|-------------------------------|
| Rust      | Build toolchain (`rustup` recommended)           | Building from source          |
| FFmpeg    | Decode video files                               | **Video playback only**       |

> FFmpeg must be available as `ffmpeg` on your `PATH`. It is **not** needed for
> static images.

---

## Installation

### Build from source

```sh
git clone https://github.com/weaver57/ASCII-Renderer.git
cd ASCII-Renderer

# Release build (recommended for real-time playback performance)
cargo build --release

# The binary is written to:
#   target/release/ascii_renderer    (Linux/macOS)
#   target\release\ascii_renderer.exe (Windows)
```

Optionally add the binary to your `PATH`:

```sh
# Linux/macOS
export PATH="$PWD/target/release:$PATH"

# Windows PowerShell
$env:PATH = "$PWD\target\release;" + $env:PATH
```

---

## Usage

Display an image:

```sh
ascii_renderer photo.png
```

Play a video (FFmpeg required):

```sh
ascii_renderer clip.mp4 --fps 30
```

A few examples:

```sh
# 80 columns wide, colored
ascii_renderer photo.png --width 80

# Grayscale, detailed ramp, inverted
ascii_renderer photo.png --color grayscale --ramp detailed --invert

# Loop a video with a custom character ramp
ascii_renderer clip.mp4 --loop-video --custom-ramp " .:|=+*#%@"
```

---

## Command-line options

| Option                   | Description                                                        | Default     |
|--------------------------|--------------------------------------------------------------------|-------------|
| `<FILE>` *(required)*    | Path to a video or image file.                                     | —           |
| `-W, --width <WIDTH>`    | Output width in character columns (auto-fits the terminal).        | auto        |
| `-H, --height <HEIGHT>`  | Output height in character rows (auto-fits the terminal).          | auto        |
| `-f, --fps <FPS>`        | Target playback frame rate for video (max 240).                    | `30`        |
| `-c, --color <COLOR>`    | Color mode: `truecolor`, `grayscale`, or `monochrome`.             | `truecolor` |
| `-r, --ramp <RAMP>`      | Ramp preset: `short`, `detailed`, or `block`.                      | `short`     |
| `--custom-ramp <STRING>` | Custom character ramp (brightness, darkest → lightest). Overrides `--ramp`. | —     |
| `-i, --invert`           | Invert the brightness → character mapping.                         | off         |
| `-l, --loop-video`       | Loop video playback continuously.                                  | off         |
| `-h, --help`             | Print help.                                                        | —           |
| `-V, --version`          | Print version.                                                     | —           |

> **Note:** `--width` / `--height` refer to *character-cell* counts, not pixels.
> When only one is given, the other is derived automatically to preserve the
> source image's aspect ratio on screen.

---

## Interactive controls

Controls work live while the renderer is running.

### Image viewer

| Key          | Action                          |
|--------------|---------------------------------|
| `q`, `Q`, `Esc` | Quit                         |
| `c`, `C`     | Cycle color mode (Truecolor → Grayscale → Monochrome) |
| `i`, `I`     | Toggle invert                  |
| `r`, `R`     | Cycle ramp preset (Short → Detailed → Block) |
| `Ctrl+C`     | Quit                           |

### Video playback

| Key              | Action                          |
|------------------|---------------------------------|
| `Space`          | Pause / resume                  |
| `q`, `Q`, `Esc`  | Quit                            |
| `c`, `C`         | Cycle color mode                |
| `i`, `I`         | Toggle invert                   |
| `r`, `R`         | Cycle ramp preset               |
| `Ctrl+C`         | Quit                            |

> If a custom ramp is set with `--custom-ramp`, it overrides the preset ramp
> selected with `r`/`R`.

---

## How it works

1. **Decode / load** — an image is decoded; a video is decoded via an FFmpeg
   subprocess streaming raw RGB frames to a pipe.
2. **Grid sizing** — output columns/rows are computed from the terminal size and
   the source aspect ratio (`CHAR_ASPECT = 0.5`) so each character cell samples a
   proportional source region and nothing gets vertically distorted.
3. **Downsample** — each source block is area-averaged (no nearest-neighbor
   aliasing).
4. **Edge detection** — Sobel runs at *full source resolution* (not the
   downsampled grid), then Non-Maximum Suppression, adaptive percentile
   double-thresholding, and queue-based hysteresis produce per-pixel edges. Edges
   are aggregated into each character cell (magnitude-weighted circular mean of
   orientation) and rendered as directional glyphs. Color stays brightness-derived.
5. **Perceptual luma** — Rec. 709 luminance (`0.2126R + 0.7152G + 0.0722B`) maps
   every non-edge cell to a ramp character.
6. **Emit** — ANSI truecolor foreground codes + glyphs for each cell, composed
   into a single buffer and written with one `flush` per frame.

---

## Project structure

```
ASCII/
├── Cargo.toml
├── assets/
│   └── ramp-font.ttf    # bundled permissively-licensed font for ramp generation
├── src/
│   ├── main.rs            # CLI entry point & orchestration
│   ├── lib.rs             # library root, module wiring
│   ├── image_loader.rs    # image loading, resize, aspect-correct grid sizing
│   ├── terminal.rs        # raw-mode / alternate-screen RAII guard
│   ├── render/
│   │   ├── mod.rs
│   │   ├── ascii.rs       # AsciiRenderer: luminance → char, edge blend, ANSI buffer
│   │   ├── edge.rs        # Sobel, NMS, hysteresis, circular-mean aggregation, dir→char
│   │   └── luminance.rs   # Rec. 709 luminance math
│   ├── bin/
│   │   └── generate_ramp.rs  # offline data-driven ramp generator (fontdue)
│   └── video/
│       ├── mod.rs
│       └── decoder.rs     # FFmpeg subprocess → raw RGB frame reader
└── tests/
    └── stress_tests.rs    # robustness stress tests
```

---

## Testing

```sh
cargo test
```

Runs unit tests plus a robustness/stress suite covering edge dimensions,
pathological inputs, Ramp permutations, UTF-8 validity, and end-to-end rendering.

---

## License

No license has been specified for this project.
