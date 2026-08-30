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
- **`--calibrate` mode** — interactive calibration to find your terminal's exact character aspect ratio, saved to `~/.ascii_renderer.toml`.
- **Area-aware downsampling** (`Triangle` filter) to avoid aliasing on fine detail.
- **Edge-aware glyphs** — structural edges detected via full-resolution Sobel + adaptive Canny (NMS, percentile thresholds, queue hysteresis) render as correctly-oriented directional characters (`|`, `-`, `/`, `\`), falling back to brightness shading everywhere else.
- **Buffered output** — each frame is composed into one buffer and flushed in a single write.
- **Live resize** — the grid automatically re-calibrates when you resize your terminal window.
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

### Quick install (recommended)

```sh
git clone https://github.com/weaver57/ASCII-Renderer.git
cd ASCII-Renderer
cargo install --path .
```

This builds a release binary and copies it to `~/.cargo/bin/`, which is already
on your `PATH` if you use `rustup`. After installation you can run it from
anywhere:

```sh
ascii_renderer photo.png
```

To update later, pull and reinstall:

```sh
git pull
cargo install --path .
```

### Build without installing

If you prefer not to install globally, you can build and run directly:

```sh
cargo build --release

# Linux/macOS
./target/release/ascii_renderer photo.png

# Windows
.\target\release\ascii_renderer.exe photo.png
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

# Use the data-driven perceptual ramp (16 chars, ink-density measured)
ascii_renderer photo.png --ramp generated
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
| `-r, --ramp <RAMP>`      | Ramp preset: `short`, `detailed`, `block`, or `generated`.         | `short`     |
| `--custom-ramp <STRING>` | Custom character ramp (brightness, darkest → lightest). Overrides `--ramp`. | —     |
| `-i, --invert`           | Invert the brightness → character mapping.                         | off         |
| `-l, --loop-video`       | Loop video playback continuously.                                  | off         |
| `--char-aspect <F>`      | Force the terminal cell aspect ratio (width/height). Overrides config/auto.  | `0.5` |
| `--calibrate`            | Interactive calibration: adjust a test circle until it looks round, then save. | off  |
| `--debug`                | Print diagnostic info (char_aspect, grid dims) to stderr before rendering.   | off  |
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
| `r`, `R`     | Cycle ramp preset (Short → Detailed → Block → Generated) |
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

## Calibration

Terminal fonts differ in character-cell proportions. The default `0.5` works for most monospace fonts, but for perfect aspect ratio:

```sh
# Run calibration once — adjust the circle until it looks round, then press Enter
ascii_renderer --calibrate

# The value is saved to ~/.ascii_renderer.toml automatically
# Override per-run with:
ascii_renderer photo.png --char-aspect 0.45
```

**Priority:** `--char-aspect` flag > `~/.ascii_renderer.toml` > default `0.5`

---

## How it works

1. **Decode / load** — an image is decoded; a video is decoded via an FFmpeg
   subprocess streaming raw RGB frames to a pipe.
2. **Grid sizing** — output columns/rows are computed from the terminal size and
   the source aspect ratio. The default cell aspect ratio is `0.5` (cells ~2× taller than wide). Run
   `--calibrate` once to find your terminal's exact value, which is saved to
   `~/.ascii_renderer.toml` for future renders. `--char-aspect` overrides per-run.
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
│   ├── config.rs          # ~/.ascii_renderer.toml read/write, calibration persistence
│   ├── render/
│   │   ├── mod.rs
│   │   ├── ascii.rs       # AsciiRenderer: luminance → char, edge blend, ANSI buffer
│   │   ├── edge.rs        # Sobel, NMS, hysteresis, circular-mean aggregation, dir→char
│   │   ├── luminance.rs   # Rec. 709 luminance math
│   │   └── ramp.rs        # generated perceptual ramp (16 chars, from fontdue ink density)
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
