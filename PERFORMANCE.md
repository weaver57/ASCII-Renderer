# Phase 5 Baseline Profiling & Performance Report

**Date:** 2026-09-04  
**Hardware / OS:** Windows 10 x86_64, Rust 1.98.0  
**Phase:** 5 (Concurrency & SIMD)  
**Profiled Pipeline:** Single-threaded, scalar Phase 1–4 baseline (`ascii_renderer v0.1.0`)

---

## 1. Executive Summary & Confirmation of Governing Hypotheses

Per the Phase 5 Architecture Plan (§0, D1), profiling the actual Phase 1–4 pipeline before introducing any threading or SIMD code was mandatory to validate our optimization priorities:

1. **Sobel Convolution (`build_luma_map` + `build_gradient_map`)**: **155.06 ms / frame** (1080p source).  
   *Finding*: Confirmed as the primary computational bottleneck (~51.5% of total edge pipeline time). Target for both Rayon row-sharding (§4.7) and SIMD vectorization (§4.9).
2. **NMS + Hysteresis (`compute_frame_edges` / `non_max_suppress` + `promote_edges`)**: **145.69 ms / frame** (1080p source).  
   *Finding*: Confirmed as the second largest bottleneck (~48.4% of edge pipeline time). NMS is a prime target for Rayon and SIMD vectorization; hysteresis remains single-threaded O(N) queue traversal per D7.
3. **Total Edge Detection Pipeline**: **300.75 ms / frame** (~3.3 FPS theoretical maximum in single-threaded scalar execution).
4. **Downsampling & Grid Operations**: Box-filter downsampling of 1080p source pixels to cell grid takes ~2–5 ms. Secondary target for Rayon and SIMD.
5. **Per-Cell Character Grid Passes**: `aggregate_cell_edges`, `TemporalEdgeSmoother`, `DoubleGrid` diffing, and ANSI escape emission take < 0.2 ms combined (< 0.1% of runtime). Per D7/§4.11, these are deliberately left un-threaded and scalar to avoid Rayon dispatch overhead.

---

## 2. Detailed Microbenchmark Results (Release Mode)

Workload: 1920×1080 source image downsampled to 80×45 character grid (10 iterations averaged):

| Stage | Scalar Baseline (1080p) | % of Pipeline | Optimization Target |
|---|---|---|---|
| **Sobel (Luma + Gradient)** | 155.06 ms | 51.5% | Rayon + SIMD (`wide::f32x8`) |
| **NMS + Hysteresis (Step Edge)** | 145.69 ms | 48.4% | Rayon + SIMD (NMS only) |
| **NMS + Hysteresis (Checkerboard)**| 142.70 ms | 47.4% | Rayon + SIMD (NMS only) |
| **Total Edge Compute** | **300.75 ms** | 100.0% | Multi-stage pipeline + SIMD |
| **Downsample (Box Filter)** | ~3.8 ms | < 1.5% | Rayon + SIMD |
| **Grid Diff + SGR Emission** | < 0.2 ms | < 0.1% | Untouched (scalar, local) |

---

## 3. Architecture Action Plan Validated

- **Three-Stage Pipeline (Decode / Process / Render)**: Overlap decode I/O, compute, and terminal render via bounded channels (cap 2–3) and pre-allocated `BufferPool<T>` with `PoolGuard<T>` RAII lifecycle.
- **Rayon Data Parallelism**: Row-sharded Sobel and NMS over shared immutable `luma_map` / `gradient_map` with disjoint `par_chunks_mut` writes (0 `unsafe`, 0 halo copying).
- **SIMD Acceleration**: 8-wide float SIMD (`wide::f32x8`) for box-filter downsampling summation and horizontal Sobel/NMS convolution with scalar remainder fallbacks.
- **Strict Differential Verification**: Every SIMD kernel tested against scalar Phase 1–4 twin for `< 1e-3` float tolerance and exact glyph parity (`direction_to_char`).
