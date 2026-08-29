mod image_loader;
mod render;
mod terminal;
mod video;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use render::{
    AsciiRenderer, ColorMode as RenderColorMode, BLOCK_RAMP, DETAILED_RAMP, RAMP, SHORT_RAMP,
    compute_frame_edges,
};
use terminal::TerminalGuard;
use video::FFmpegDecoder;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorChoice {
    Truecolor,
    Grayscale,
    Monochrome,
}

impl From<ColorChoice> for RenderColorMode {
    fn from(c: ColorChoice) -> Self {
        match c {
            ColorChoice::Truecolor => RenderColorMode::TrueColor,
            ColorChoice::Grayscale => RenderColorMode::Grayscale,
            ColorChoice::Monochrome => RenderColorMode::Monochrome,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum RampChoice {
    Short,
    Detailed,
    Block,
    Generated,
}

impl RampChoice {
    pub fn as_str(&self) -> &'static str {
        match self {
            RampChoice::Short => SHORT_RAMP,
            RampChoice::Detailed => DETAILED_RAMP,
            RampChoice::Block => BLOCK_RAMP,
            RampChoice::Generated => "",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            RampChoice::Short => RampChoice::Detailed,
            RampChoice::Detailed => RampChoice::Block,
            RampChoice::Block => RampChoice::Generated,
            RampChoice::Generated => RampChoice::Short,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "High-performance RGB ASCII Video/Image Renderer in Rust", long_about = None)]
struct Args {
    /// Path to video or image file
    #[arg(value_name = "FILE")]
    input: PathBuf,

    /// Output width in character columns (default: auto-fit terminal)
    #[arg(short = 'W', long)]
    width: Option<usize>,

    /// Output height in character rows (default: auto-fit terminal)
    #[arg(short = 'H', long)]
    height: Option<usize>,

    /// Target playback frame rate for video
    #[arg(short, long, default_value_t = 30.0)]
    fps: f64,

    /// Color rendering mode
    #[arg(short, long, value_enum, default_value_t = ColorChoice::Truecolor)]
    color: ColorChoice,

    /// Preset ASCII character ramp
    #[arg(short, long, value_enum, default_value_t = RampChoice::Short)]
    ramp: RampChoice,

    /// Custom character ramp string (overrides --ramp preset)
    #[arg(long)]
    custom_ramp: Option<String>,

    /// Invert character brightness mapping
    #[arg(short, long, default_value_t = false)]
    invert: bool,

    /// Loop video playback continuously
    #[arg(short = 'l', long, default_value_t = false)]
    loop_video: bool,
}

fn is_image_extension(path: &std::path::Path) -> bool {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        matches!(
            ext.to_lowercase().as_str(),
            "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif"
        )
    } else {
        false
    }
}

fn determine_dimensions(
    custom_w: Option<usize>,
    custom_h: Option<usize>,
) -> Result<(usize, usize)> {
    let (term_cols, term_rows) = TerminalGuard::get_size().context("Failed to get terminal size")?;

    // Reserve 1 line at the bottom for status / clean terminal margins
    let max_w = term_cols as usize;
    let max_h = (term_rows.saturating_sub(1)) as usize;

    let w = custom_w.unwrap_or(max_w).min(max_w).max(1);
    let h = custom_h.unwrap_or(max_h).min(max_h).max(1);

    Ok((w, h))
}

fn sanitize_fps(fps: f64) -> f64 {
    if fps.is_finite() && fps > 0.0 && fps <= 240.0 {
        fps
    } else {
        30.0
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.input.exists() {
        anyhow::bail!("Input file does not exist: {:?}", args.input);
    }

    let target_fps = sanitize_fps(args.fps);
    let mut current_ramp_preset = args.ramp;
    let custom_ramp_opt = args.custom_ramp.clone();
    let mut color_mode: RenderColorMode = args.color.into();
    let mut invert = args.invert;

    let get_current_ramp = |preset: RampChoice, custom: &Option<String>| -> String {
        if let Some(c) = custom {
            c.clone()
        } else {
            match preset {
                RampChoice::Generated => RAMP.iter().collect::<String>(),
                _ => preset.as_str().to_string(),
            }
        }
    };

    // Static Image Interactive Viewer
    if is_image_extension(&args.input) {
        let _guard = TerminalGuard::init()?;

        // Read just the image header (cheap) so we can preserve aspect ratio
        // when mapping the image onto the character grid.
        let (img_w, img_h) = image::image_dimensions(&args.input)
            .with_context(|| format!("Failed to read dimensions of image at {:?}", args.input))?;
        let (term_cols, term_rows) = TerminalGuard::get_size().context("Failed to get terminal size")?;
        let (width, height) = image_loader::compute_image_grid_dimensions(
            img_w,
            img_h,
            args.width,
            args.height,
            term_cols,
            term_rows,
        );

        let image_frame = image_loader::load_and_resize_image(&args.input, width as u32, height as u32)?;
        let mut output_buf = Vec::with_capacity(width * height * 24);
        let stdout = io::stdout();
        let mut writer = BufWriter::with_capacity(output_buf.len() + 1024, stdout.lock());

        // Full-resolution source for Phase 2 edge detection. Sobel runs at the
        // native pixel grid (not the downsampled cell grid), then the per-cell
        // edge classifications drive the directional glyphs. Re-decoded once
        // here; the glyph set itself is recomputed on each interactive keypress
        // but never the convolution.
        let full_frame = image_loader::load_rgb_frame(&args.input)?;
        let cell_edges = compute_frame_edges(
            &full_frame.rgb_data,
            full_frame.width as usize,
            full_frame.height as usize,
            width,
            height,
        );

        let mut needs_redraw = true;

        loop {
            if needs_redraw {
                let ramp_str = get_current_ramp(current_ramp_preset, &custom_ramp_opt);
                let renderer = AsciiRenderer::new(&ramp_str, color_mode, invert);
                renderer.render_frame_with_edges(
                    &image_frame.rgb_data,
                    width,
                    height,
                    &cell_edges,
                    &mut output_buf,
                );
                writer.write_all(&output_buf)?;
                writer.flush()?;
                needs_redraw = false;
            }

            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                            break;
                        }
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break,
                            KeyCode::Char('c') | KeyCode::Char('C') => {
                                color_mode = match color_mode {
                                    RenderColorMode::TrueColor => RenderColorMode::Grayscale,
                                    RenderColorMode::Grayscale => RenderColorMode::Monochrome,
                                    RenderColorMode::Monochrome => RenderColorMode::TrueColor,
                                };
                                needs_redraw = true;
                            }
                            KeyCode::Char('i') | KeyCode::Char('I') => {
                                invert = !invert;
                                needs_redraw = true;
                            }
                            KeyCode::Char('r') | KeyCode::Char('R') => {
                                current_ramp_preset = current_ramp_preset.next();
                                needs_redraw = true;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        return Ok(());
    }

    // Video Playback Loop
    let _guard = TerminalGuard::init()?;
    let (width, height) = determine_dimensions(args.width, args.height)?;

    let frame_duration = Duration::from_secs_f64(1.0 / target_fps);
    let mut raw_frame = vec![0u8; width * height * 3];
    let mut output_buf = Vec::with_capacity(width * height * 24);

    let stdout = io::stdout();
    let mut writer = BufWriter::with_capacity(128 * 1024, stdout.lock());

    'outer: loop {
        let mut decoder = FFmpegDecoder::new(&args.input, width, height, Some(target_fps))?;
        let ramp_str = get_current_ramp(current_ramp_preset, &custom_ramp_opt);
        let mut renderer = AsciiRenderer::new(&ramp_str, color_mode, invert);
        let mut paused = false;

        while decoder.read_frame(&mut raw_frame)? {
            let frame_start = Instant::now();

            // Handle keyboard controls
            while event::poll(Duration::from_millis(0))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                            break 'outer;
                        }
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break 'outer,
                            KeyCode::Char(' ') => paused = !paused,
                            KeyCode::Char('c') | KeyCode::Char('C') => {
                                color_mode = match color_mode {
                                    RenderColorMode::TrueColor => RenderColorMode::Grayscale,
                                    RenderColorMode::Grayscale => RenderColorMode::Monochrome,
                                    RenderColorMode::Monochrome => RenderColorMode::TrueColor,
                                };
                                let ramp = get_current_ramp(current_ramp_preset, &custom_ramp_opt);
                                renderer = AsciiRenderer::new(&ramp, color_mode, invert);
                            }
                            KeyCode::Char('i') | KeyCode::Char('I') => {
                                invert = !invert;
                                let ramp = get_current_ramp(current_ramp_preset, &custom_ramp_opt);
                                renderer = AsciiRenderer::new(&ramp, color_mode, invert);
                            }
                            KeyCode::Char('r') | KeyCode::Char('R') => {
                                current_ramp_preset = current_ramp_preset.next();
                                let ramp = get_current_ramp(current_ramp_preset, &custom_ramp_opt);
                                renderer = AsciiRenderer::new(&ramp, color_mode, invert);
                            }
                            _ => {}
                        }
                    }
                }
            }

            while paused {
                if event::poll(Duration::from_millis(50))? {
                    if let Event::Key(key) = event::read()? {
                        if key.kind == KeyEventKind::Press {
                            if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                                break 'outer;
                            }
                            match key.code {
                                KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break 'outer,
                                KeyCode::Char(' ') => paused = false,
                                _ => {}
                            }
                        }
                    }
                }
            }

            renderer.render_frame(&raw_frame, width, height, &mut output_buf);
            writer.write_all(&output_buf)?;
            writer.flush()?;

            // Frame pacing
            let elapsed = frame_start.elapsed();
            if elapsed < frame_duration {
                std::thread::sleep(frame_duration - elapsed);
            }
        }

        if !args.loop_video {
            break;
        }
    }

    Ok(())
}
