use ascii_renderer::image_loader::load_and_resize_image;
use ascii_renderer::render::{
    AsciiRenderer, ColorMode, BLOCK_RAMP, DETAILED_RAMP, SHORT_RAMP,
};
use ascii_renderer::video::FFmpegDecoder;
use std::time::Instant;

#[test]
fn test_stress_ramp_permutations() {
    let test_ramps = [
        "",
        " ",
        "@",
        " .:-=+*#%@",
        BLOCK_RAMP,
        DETAILED_RAMP,
        "░▒▓█",
        "😀😃😄😁😆",
        "アイウエオ",
        "1234567890!@#$%^&*()",
    ];

    for ramp in test_ramps {
        for invert in [false, true] {
            for mode in [ColorMode::TrueColor, ColorMode::Grayscale, ColorMode::Monochrome] {
                let renderer = AsciiRenderer::new(ramp, mode, invert);
                for lum in 0..=255 {
                    let ch = renderer.luminance_to_char(lum);
                    assert!(!ch.is_control());
                }
            }
        }
    }
}

#[test]
fn test_stress_extreme_dimensions_and_buffers() {
    let renderer = AsciiRenderer::new(SHORT_RAMP, ColorMode::TrueColor, false);
    let mut out = Vec::new();

    // 0x0 dimension
    renderer.render_frame(&[], 0, 0, &mut out);
    assert!(out.is_empty());

    // 1x1 dimension
    let pixel = [128, 64, 32];
    renderer.render_frame(&pixel, 1, 1, &mut out);
    assert!(!out.is_empty());

    // 1x500 dimension
    let col_pixels = vec![200u8; 1500];
    renderer.render_frame(&col_pixels, 1, 500, &mut out);
    assert!(!out.is_empty());

    // 500x1 dimension
    let row_pixels = vec![100u8; 1500];
    renderer.render_frame(&row_pixels, 500, 1, &mut out);
    assert!(!out.is_empty());

    // Buffer smaller than width * height * 3 (truncated data) -> must not panic
    let truncated = vec![50u8; 10];
    out.clear();
    renderer.render_frame(&truncated, 100, 100, &mut out);
    assert!(out.is_empty());
}

#[test]
fn test_stress_color_switching_throughput() {
    let width = 160;
    let height = 60;
    let frame_size = width * height * 3;

    // Generate gradient pattern
    let mut frame = vec![0u8; frame_size];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            frame[idx] = ((x * 255) / width) as u8;
            frame[idx + 1] = ((y * 255) / height) as u8;
            frame[idx + 2] = (((x + y) * 255) / (width + height)) as u8;
        }
    }

    let renderer = AsciiRenderer::new(SHORT_RAMP, ColorMode::TrueColor, false);
    let mut out = Vec::with_capacity(width * height * 32);

    let start = Instant::now();
    let iterations = 2000;
    for _ in 0..iterations {
        renderer.render_frame(&frame, width, height, &mut out);
    }
    let elapsed = start.elapsed();
    let fps = (iterations as f64) / elapsed.as_secs_f64();

    println!(
        "Rendered {} frames of {}x{} in {:?} ({:.2} FPS)",
        iterations, width, height, elapsed, fps
    );
    assert!(fps > 100.0, "Renderer must achieve at least 100 FPS (got {:.2})", fps);
}

#[test]
fn test_stress_all_color_modes_utf8_validity() {
    let width = 64;
    let height = 32;
    let mut frame = vec![0u8; width * height * 3];

    // Fill with semi-random RGB values
    for (i, byte) in frame.iter_mut().enumerate() {
        *byte = ((i * 37 + 13) % 256) as u8;
    }

    let ramps = [SHORT_RAMP, DETAILED_RAMP, BLOCK_RAMP];
    let modes = [ColorMode::TrueColor, ColorMode::Grayscale, ColorMode::Monochrome];

    let mut out = Vec::new();
    for ramp in ramps {
        for mode in modes {
            for invert in [false, true] {
                let renderer = AsciiRenderer::new(ramp, mode, invert);
                renderer.render_frame(&frame, width, height, &mut out);

                let text = String::from_utf8(out.clone());
                assert!(text.is_ok(), "Rendered output must always be valid UTF-8");
            }
        }
    }
}

#[test]
fn test_stress_decoder_invalid_files() {
    let invalid_res = FFmpegDecoder::new("this_file_does_not_exist.mp4", 64, 32, None);
    assert!(invalid_res.is_err());

    let invalid_dims = FFmpegDecoder::new("Cargo.toml", 0, 0, None);
    assert!(invalid_dims.is_err());
}

#[test]
fn test_stress_image_loader_invalid_files() {
    let invalid_res = load_and_resize_image("nonexistent_img.png", 64, 32);
    assert!(invalid_res.is_err());
}
