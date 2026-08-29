# Project: Smart RGB ASCII Video Renderer

## Context / Motivation
I want to build a real-time video-to-ASCII renderer in a low-level language (Rust or C), as a project to learn fundamental systems-programming concepts (memory management, buffers, performance optimization, SIMD, unsafe code) by applying them to something visual and immediately satisfying.

**Inspiration:** Saw a similar project built in Python + FFmpeg + ANSI True Color that converts video frames into real-time colored ASCII art in the terminal, with audio playback. I want to build my own version, but go further — treat it as a genuine rendering *algorithm*, not just a pixel-to-character lookup, and build it in a lower-level language for the performance and learning value.

## Core Pipeline
1. Decode video frames (likely via FFmpeg or a Rust/C video decoding library)
2. Convert each frame's pixels into ASCII characters with color
3. Write output to the terminal using ANSI escape codes (buffered, real-time)
4. (Optional) sync with audio playback

## Algorithmic Enhancements (this is what makes it original, not just a converter)

### 1. Basic rendering (baseline)
`pixel → brightness → character`
Map pixel brightness to a fixed ramp of characters (e.g. `@#*+:. `), darkest to lightest.

### 2. Edge-based rendering (Sobel/gradient)
```
pixel neighborhood → Sobel/gradient calculation → directional ASCII edges
```
Look at each pixel's neighbors to detect the direction of the strongest brightness change (an edge). Map detected edge direction to a matching character:
- horizontal edge → `-`
- vertical edge → `|`
- diagonal edges → `/` or `\`

This makes shapes and outlines pop instead of the image looking like fuzzy gray blobs. It's also a great low-level target: Sobel is a small matrix convolution per pixel — a classic use case for SIMD instructions in Rust/C, so this is where hardware-level optimization becomes visible as a real speedup.

### 3. Adaptive character selection
Instead of one global brightness-to-character mapping for the whole frame, choose characters based on **local structure**:
- Where Sobel detects a strong edge → use directional edge characters
- Where the area is flat/smooth → fall back to brightness-based characters

This is essentially combining #1 and #2 intelligently — cheap to add once edge detection exists, but a big visual improvement.

### 4. Temporal / motion-aware rendering
```
Frame N → Frame N+1 → calculate movement → motion-aware character rendering
```
Full optical flow / motion tracking is research-level and best avoided for a first pass. A simpler, high-value version: **frame diffing** — compare frame N to frame N-1, and only re-render terminal regions that actually changed (a "dirty rectangle" technique used in old game engines and terminal UIs). This both teaches a real performance pattern and makes the renderer noticeably faster/smoother, since most of a video frame is usually static between frames.

## Suggested Build Order
1. Get basic brightness → character mapping working end to end (baseline that produces visible output)
2. Add Sobel edge detection, map edges to directional characters
3. Blend brightness + edge detection into adaptive character selection
4. Add frame-diffing so only changed regions are redrawn

Each step produces a visible, working improvement — no long stretches of building blind.

## Why This Teaches Low-Level Fundamentals
- Manual frame buffer / pixel array management
- Sobel convolution as a SIMD optimization target (AVX2 etc.)
- Buffered, low-latency terminal I/O (writing ANSI codes efficiently)
- Frame-diffing as a real-world performance/memory-access pattern
- (If extended) audio sync introduces real-time timing constraints

## Reference
Original inspiration project (Python + FFmpeg + ANSI True Color, with audio playback):
https://github.com/RipperdocNiladri/ASCII-Art.git
