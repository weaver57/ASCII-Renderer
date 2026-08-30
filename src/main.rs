mod image_loader;
mod render;
mod terminal;
mod terminal_size;
mod config;
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

    /// Override terminal cell aspect ratio (width/height)
    ///
    /// By default, the renderer queries your terminal's actual cell dimensions
    /// and computes the correct aspect ratio. Use this flag to force a specific
    /// value (e.g., 0.45 for a tall-narrow font, 0.6 for a wide font).
    /// A value of 0.5 means cells are exactly 2x taller than wide.
    #[arg(long)]
    char_aspect: Option<f32>,

    /// Print diagnostic info to stderr before rendering
    #[arg(long, default_value_t = false)]
    debug: bool,

    /// Interactive calibration mode: shows a test circle and guides you to find your terminal's correct char_aspect
    #[arg(long, default_value_t = false)]
    calibrate: bool,
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

fn sanitize_fps(fps: f64) -> f64 {
    if fps.is_finite() && fps > 0.0 && fps <= 240.0 {
        fps
    } else {
        30.0
    }
}


/// Interactive calibration: renders a circle at varying char_aspect values
/// so the user can visually find the one that makes it look perfectly round.
/// Saves the chosen value to ~/.ascii_renderer.toml.

fn run_calibration(_debug: bool) -> Result<()> {
    // Use test_circle.png if available, otherwise generate one
    let circle_path = std::path::PathBuf::from("test_circle.png");
    let (img_w, img_h, input_path, _temp_dir);

    if circle_path.exists() {
        let dims = image::image_dimensions(&circle_path)
            .context("Failed to read test_circle.png dimensions")?;
        (img_w, img_h) = dims;
        input_path = circle_path;
        _temp_dir = None;
    } else {
        let dir = std::env::temp_dir().join("ascii_calibrate");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("calibrate_circle.png");
        let size = 200u32;
        let mut img = image::RgbImage::new(size, size);
        let center = size as f32 / 2.0;
        let radius = size as f32 / 2.5;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= radius {
                    img.put_pixel(x, y, image::Rgb([255u8, 255, 255]));
                } else if dist <= radius + 2.0 {
                    let t = ((radius + 2.0 - dist) / 2.0).clamp(0.0, 1.0);
                    let v = (t * 255.0) as u8;
                    img.put_pixel(x, y, image::Rgb([v, v, v]));
                } else {
                    img.put_pixel(x, y, image::Rgb([0u8, 0, 0]));
                }
            }
        }
        img.save(&path)?;
        (img_w, img_h) = (size, size);
        input_path = path;
        _temp_dir = Some(dir);
    }

    let _guard = terminal::TerminalGuard::init()?;

    let (term_cols, term_rows) =
        crossterm::terminal::size().context("Failed to get terminal size")?;

    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());

    let mut current_aspect: f32 = 0.50;
    let step: f32 = 0.01;

    eprintln!("=== CALIBRATION MODE ===");
    eprintln!("The circle should look perfectly round.");
    eprintln!("LEFT/RIGHT: adjust by 0.01 | UP/DOWN: adjust by 0.05");
    eprintln!("ENTER: save | ESC: cancel");
    eprintln!("========================");

    loop {
        let (width, height) = image_loader::compute_image_grid_dimensions(
            img_w, img_h, Some(term_cols as usize), None,
            term_cols, term_rows, current_aspect,
        );

        let image_frame = image_loader::load_and_resize_image(
            &input_path, width as u32, height as u32,
        )?;

        let full_frame = image_loader::load_rgb_frame(&input_path)?;
        let cell_edges = render::compute_frame_edges(
            &full_frame.rgb_data,
            full_frame.width as usize,
            full_frame.height as usize,
            width,
            height,
        );

        let mut output_buf = Vec::with_capacity(width * height * 24);
        let renderer = render::AsciiRenderer::new(
            render::SHORT_RAMP,
            render::ColorMode::TrueColor,
            false,
        );
        renderer.render_frame_with_edges(
            &image_frame.rgb_data, width, height, &cell_edges, &mut output_buf,
        );
        writer.write_all(&output_buf)?;

        let status = format!(
            "  char_aspect = {:.2}  [grid {}x{}]  <- adjust | ENTER=save | ESC=cancel  ",
            current_aspect, width, height
        );
        write!(writer, "\r\n\x1b[0m{}", status)?;
        writer.flush()?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Esc | KeyCode::Char('q') => {
                            return Ok(());
                        }
                        KeyCode::Enter => {
                            config::Config::save_char_aspect(current_aspect)?;
                            eprintln!("\r\n\r\nCalibration complete! Your char_aspect = {:.2} has been saved.", current_aspect);
                            eprintln!("Future renders will use this value automatically.");
                            eprintln!("Override anytime with: --char-aspect <value>");
                            return Ok(());
                        }
                        KeyCode::Left | KeyCode::Char('h') => {
                            current_aspect = (current_aspect - step).max(0.10);
                        }
                        KeyCode::Right | KeyCode::Char('l') => {
                            current_aspect = (current_aspect + step).min(1.00);
                        }
                        KeyCode::Up => {
                            current_aspect = (current_aspect + 0.05).min(1.00);
                        }
                        KeyCode::Down => {
                            current_aspect = (current_aspect - 0.05).max(0.10);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Apply char aspect override before any rendering operations
    // Priority: --char-aspect flag > config file > hardcoded default
    if let Some(aspect) = args.char_aspect {
        terminal_size::set_char_aspect_override(aspect);
    } else {
        // Load from config file if available
        let cfg = config::Config::load();
        if let Some(aspect) = cfg.char_aspect {
            terminal_size::set_char_aspect_override(aspect);
        }
    }

    // Handle --calibrate mode (runs before any image/video rendering)
    if args.calibrate {
        return run_calibration(args.debug);
    }

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

    // ── Static Image Interactive Viewer ──────────────────────────────────
    if is_image_extension(&args.input) {
        let (img_w, img_h) = image::image_dimensions(&args.input)
            .with_context(|| format!("Failed to read dimensions of image at {:?}", args.input))?;
        let char_aspect = terminal_size::get_char_aspect();

        let _guard = TerminalGuard::init()?;

        let (mut term_cols, mut term_rows) =
            crossterm::terminal::size().context("Failed to get terminal size")?;
        let (mut width, mut height) = image_loader::compute_image_grid_dimensions(
            img_w, img_h, args.width, args.height, term_cols, term_rows, char_aspect,
        );
        let mut old_w = width;
        let mut old_h = height;
        if args.debug {
            eprintln!("[DEBUG] char_aspect = {:.4}", char_aspect);
            eprintln!("[DEBUG] terminal = {} cols x {} rows", term_cols, term_rows);
            eprintln!("[DEBUG] image = {}x{} px", img_w, img_h);
            eprintln!("[DEBUG] grid = {} cols x {} rows", width, height);
            eprintln!(
                "[DEBUG] ramp = {:?}",
                get_current_ramp(current_ramp_preset, &custom_ramp_opt)
                    .chars().collect::<Vec<_>>()
            );
            eprintln!("[DEBUG] color_mode = {:?}", color_mode);
            eprintln!("[DEBUG] invert = {}", invert);
        }

        let mut image_frame =
            image_loader::load_and_resize_image(&args.input, width as u32, height as u32)?;
        let mut output_buf = Vec::with_capacity(width * height * 24);
        let stdout = io::stdout();
        let mut writer = BufWriter::with_capacity(output_buf.len() + 1024, stdout.lock());

        let full_frame = image_loader::load_rgb_frame(&args.input)?;
        let mut cell_edges = compute_frame_edges(
            &full_frame.rgb_data,
            full_frame.width as usize,
            full_frame.height as usize,
            width,
            height,
        );

        let mut needs_redraw = true;

        loop {
            // Detect terminal resize
            if let Ok((new_cols, new_rows)) = crossterm::terminal::size() {
                if new_cols != term_cols || new_rows != term_rows {
                    term_cols = new_cols;
                    term_rows = new_rows;
                    let (new_w, new_h) = image_loader::compute_image_grid_dimensions(
                        img_w, img_h, args.width, args.height,
                        term_cols, term_rows, char_aspect,
                    );
                    old_w = width;
                    old_h = height;
                    width = new_w;
                    height = new_h;
                    image_frame = image_loader::load_and_resize_image(
                        &args.input, width as u32, height as u32,
                    )?;
                    cell_edges = compute_frame_edges(
                        &full_frame.rgb_data, full_frame.width as usize,
                        full_frame.height as usize, width, height,
                    );
                    output_buf = Vec::with_capacity(width * height * 24);
                    needs_redraw = true;
                }
            }

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
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && key.code == KeyCode::Char('c')
                        {
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

    // ── Video Playback Loop ──────────────────────────────────────────────
    let (term_cols, term_rows) =
        crossterm::terminal::size().context("Failed to get terminal size")?;

    let (src_w, src_h) = video::probe_video_dimensions(&args.input).unwrap_or_else(|| {
        eprintln!(
            "[ascii_renderer] warning: could not probe video dimensions for {:?}. \
             Falling back to 16:9.",
            args.input
        );
        (1920, 1080)
    });

    let char_aspect = terminal_size::get_char_aspect();
    let (width, height) = image_loader::compute_image_grid_dimensions(
        src_w,
        src_h,
        args.width,
        args.height,
        term_cols,
        term_rows,
        char_aspect,
    );

    if args.debug {
        eprintln!("[DEBUG] char_aspect = {:.4}", char_aspect);
        eprintln!(
            "[DEBUG] terminal = {} cols x {} rows",
            term_cols, term_rows
        );
        eprintln!("[DEBUG] video source = {}x{}", src_w, src_h);
        eprintln!("[DEBUG] grid = {} cols x {} rows", width, height);
    }

    let _guard = TerminalGuard::init()?;

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
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && key.code == KeyCode::Char('c')
                        {
                            break 'outer;
                        }
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                                break 'outer
                            }
                            KeyCode::Char(' ') => paused = !paused,
                            KeyCode::Char('c') | KeyCode::Char('C') => {
                                color_mode = match color_mode {
                                    RenderColorMode::TrueColor => RenderColorMode::Grayscale,
                                    RenderColorMode::Grayscale => RenderColorMode::Monochrome,
                                    RenderColorMode::Monochrome => RenderColorMode::TrueColor,
                                };
                                let ramp =
                                    get_current_ramp(current_ramp_preset, &custom_ramp_opt);
                                renderer = AsciiRenderer::new(&ramp, color_mode, invert);
                            }
                            KeyCode::Char('i') | KeyCode::Char('I') => {
                                invert = !invert;
                                let ramp =
                                    get_current_ramp(current_ramp_preset, &custom_ramp_opt);
                                renderer = AsciiRenderer::new(&ramp, color_mode, invert);
                            }
                            KeyCode::Char('r') | KeyCode::Char('R') => {
                                current_ramp_preset = current_ramp_preset.next();
                                let ramp =
                                    get_current_ramp(current_ramp_preset, &custom_ramp_opt);
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
                            if key.modifiers.contains(KeyModifiers::CONTROL)
                                && key.code == KeyCode::Char('c')
                            {
                                break 'outer;
                            }
                            match key.code {
                                KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                                    break 'outer
                                }
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
