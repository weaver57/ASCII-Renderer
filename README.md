# ASCII Renderer

A high-performance RGB ASCII **video and image renderer** written in Rust. It
decodes video frames via FFmpeg or loads static images, downsamples each frame
to a character grid, and streams colored ASCII art to your terminal in real time
using ANSI truecolor escape codes — all through a single buffered write per frame.

```
Usage: ascii_renderer [OPTIONS] <FILE>
```

---

## Features

- **Static image viewer** with live interactive controls (color mode, ramp, invert).
- **Real-time video playback** (via FFmpeg subprocess) with pause/resume and loop.
- **YUV-native video pipeline** — decodes to YUV420P directly, uses the Y plane as
  the luma map for edge detection (no RGB conversion in the hot path).
- **Perceptual luminance** (Rec. 709 coefficients) for true-to-eye brightness mapping.
- **Aspect-ratio-correct grid sizing** — output is never vertically stretched or squashed.
- **`--calibrate` mode** — interactive calibration to find your terminal's exact character
  aspect ratio, saved to `~/.ascii_renderer.toml`.
- **Area-aware downsampling** (Triangle filter) to avoid aliasing on fine detail.
- **Edge-aware glyphs** — structural edges detected via full-resolution Sobel + adaptive
  Canny (NMS, percentile thresholds, queue hysteresis) render as correctly-oriented
  directional characters (`|`, `-`, `/`, `\`), falling back to brightness shading everywhere else.
- **Temporal edge smoothing** — Phase 2's `TemporalEdgeSmoother` is wired into the video
  loop, stabilizing flickering edges across consecutive frames.
- **Buffered output** — each frame is composed into one buffer and flushed in a single write.
- **Live resize** — the grid automatically re-calibrates when you resize your terminal window.
- **Clock-based frame pacing** — PTS-aware playback with drop-frame policy for slow machines.
- Three color modes and multiple adjustable ASCII ramps, switchable mid-run.

---

## Requirements

| Tool      | Purpose                              | Required for          |
|-----------|--------------------------------------|-----------------------|
| Rust      | Build toolchain (`rustup` recommended) | Building from source |
| FFmpeg    | Decode video files                   | **Video playback**    |

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

More examples:

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

| Option                     | Description                                                                       | Default     |
|----------------------------|-----------------------------------------------------------------------------------|-------------|
| `<FILE>` *(required)*     | Path to a video or image file.                                                    | —           |
| `-W, --width <WIDTH>`     | Output width in character columns (auto-fits the terminal).                       | auto        |
| `-H, --height <HEIGHT>`   | Output height in character rows (auto-fits the terminal).                         | auto        |
| `-f, --fps <FPS>`         | Target playback frame rate for video (max 240).                                   | `30`        |
| `-c, --color <COLOR>`     | Color mode: `truecolor`, `grayscale`, or `monochrome`.                            | `truecolor` |
| `-r, --ramp <RAMP>`       | Ramp preset: `short`, `detailed`, `block`, or `generated`.                        | `short`     |
| `--custom-ramp <STRING>`  | Custom character ramp (darkest → lightest). Overrides `--ramp`.                   | —           |
| `-i, --invert`            | Invert the brightness → character mapping.                                        | off         |
| `-l, --loop-video`        | Loop video playback continuously.                                                 | off         |
| `--char-aspect <F>`       | Override terminal cell aspect ratio (width/height). Overrides config/auto.        | `0.5`       |
| `--calibrate`             | Interactive calibration: adjust a test circle until it looks round, then save.    | off         |
| `--debug`                 | Print diagnostic info (char_aspect, grid dims) to stderr before rendering.        | off         |
| `-h, --help`              | Print help.                                                                       | —           |
| `-V, --version`           | Print version.                                                                    | —           |

> **Note:** `--width` / `--height` refer to *character-cell* counts, not pixels.
> When only one is given, the other is derived automatically to preserve the
> source's aspect ratio on screen.

---

## Interactive controls

Controls work live while the renderer is running.

### Image viewer

| Key              | Action                                              |
|------------------|-----------------------------------------------------|
| `q`, `Q`, `Esc` | Quit                                                |
| `c`, `C`         | Cycle color mode (Truecolor → Grayscale → Monochrome) |
| `i`, `I`         | Toggle invert                                       |
| `r`, `R`         | Cycle ramp preset (Short → Detailed → Block → Generated) |
| `Ctrl+C`         | Quit                                                |

### Video playback

| Key              | Action                                              |
|------------------|-----------------------------------------------------|
| `Space`          | Pause / resume                                      |
| `q`, `Q`, `Esc`  | Quit                                                |
| `c`, `C`         | Cycle color mode                                    |
| `i`, `I`         | Toggle invert                                       |
| `r`, `R`         | Cycle ramp preset                                   |
| `Ctrl+C`         | Quit                                                |

> If a custom ramp is set with `--custom-ramp`, it overrides the preset ramp
> selected with `r`/`R`.

---

## Calibration

Terminal fonts differ in character-cell proportions. The default `0.5` works for
most monospace fonts, but for perfect aspect ratio:

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

### Image pipeline

1. **Load** — image decoded via the `image` crate at native resolution.
2. **Grid sizing** — output columns/rows computed from terminal size and source
   aspect ratio, using the configured character aspect ratio.
3. **Downsample** — each source block is area-averaged (Triangle filter).
4. **Edge detection** — Sobel at *full source resolution*, then NMS, adaptive
   percentile double-thresholding, and queue-based hysteresis. Edges are
   aggregated per cell (magnitude-weighted circular mean of orientation) and
   rendered as directional glyphs.
5. **Perceptual luma** — Rec. 709 luminance maps non-edge cells to ramp characters.
6. **Emit** — ANSI truecolor codes + glyphs, composed into one buffer, single flush.

### Video pipeline

1. **Decode** — FFmpeg subprocess streams raw YUV420P frames to a pipe.
2. **De-stride** — FFmpeg's row padding is stripped into tightly-packed buffers.
3. **Range expansion** — limited-range (16–235) expanded to full-range (0–255).
4. **Luma map** — Y plane used directly as the brightness/edge-detection source
   (no RGB conversion in the hot path).
5. **Edge detection** — same Sobel/NMS/hysteresis chain as images, operating on
   the Y-plane luma map at full source resolution.
6. **Temporal smoothing** — `TemporalEdgeSmoother` blends edge state across frames
   to reduce flicker.
7. **Downsample** — box-filter averages Y, U, V per cell; YUV→RGB converted once
   per cell (not per pixel) for ~778× fewer color conversions at 1080p→160×90.
8. **Pace** — PTS-based clock with drop-frame policy: frames more than 2 frame-
   durations behind schedule are skipped entirely to recover lag.
9. **Emit** — cursor-home (`\x1b[H`) before each frame so it overwrites in place.

---

## Project structure

```
ASCII-Renderer/
├── Cargo.toml
├── README.md
├── assets/
│   └── ramp-font.ttf              # bundled font for ramp generation
├── src/
│   ├── main.rs                    # CLI entry point & orchestration
│   ├── lib.rs                     # library root, module wiring
│   ├── config.rs                  # ~/.ascii_renderer.toml read/write
│   ├── image_loader.rs            # image loading, resize, grid sizing
│   ├── terminal.rs                # raw-mode / alternate-screen RAII guard
│   ├── terminal_size.rs           # char-aspect detection & override
│   ├── render/
│   │   ├── mod.rs
│   │   ├── ascii.rs               # luminance→char, edge blend, ANSI buffer
│   │   ├── edge.rs                # Sobel, NMS, hysteresis, dir→char
│   │   ├── luminance.rs           # Rec. 709 luminance math
│   │   ├── ramp.rs                # perceptual ramp (ink-density measured)
│   │   └── temporal.rs            # cross-frame edge smoothing
│   ├── bin/
│   │   ├── benchmark.rs           # decode/render throughput benchmarks
│   │   └── generate_ramp.rs       # offline ramp generator (fontdue)
│   └── video/
│       ├── mod.rs
│       ├── decoder.rs             # FFmpeg subprocess → raw frame reader
│       ├── probe.rs               # video dimension probing via ffprobe
│       └── yuv.rs                 # YUV420P de-striding, range, YUV→RGB
└── tests/
    └── stress_tests.rs            # robustness stress tests
```

---

## Testing

```sh
cargo test
```

Runs unit tests plus a robustness/stress suite covering edge dimensions,
pathological inputs, ramp permutations, UTF-8 validity, YUV conversion
correctness, and end-to-end rendering.

---

## License

No license has been specified for this project.
