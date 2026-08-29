// Phase 2 — Data-Driven Glyph Ramp Generator
//
// Measures ink density of candidate glyphs using a bundled monospace font,
// then selects N evenly-spaced density targets for a perceptually-uniform ramp.
//
// Usage: cargo run --bin generate_ramp [--ramp-size N] [--font-size S]
//
// Determinism: same font + charset + params always produces the same output
// (no HashMap iteration, deterministic tie-breaking).

fn main() {
    // 1. Embed the reference font (bundled at compile time)
    const FONT_BYTES: &[u8] = include_bytes!("../../assets/ramp-font.ttf");
    let font = fontdue::Font::from_bytes(FONT_BYTES, fontdue::FontSettings::default())
        .expect("Failed to load bundled font");

    // 2. Candidate character set — broad superset spanning sparse to dense
    let candidates: &[char] = &[
        ' ', '.', '\'', '`', '^', '"', ',', ':', ';', 'I', 'l', '!', 'i', '>', '<', '~', '+',
        '_', '-', '?', ']', '[', '}', '{', '1', ')', '(', '|', '\\', '/', 't', 'f', 'j', 'r',
        'x', 'n', 'u', 'v', 'c', 'z', 'X', 'Y', 'U', 'J', 'C', 'L', 'Q', '0', 'O', 'Z', 'm',
        'w', 'q', 'p', 'd', 'b', 'k', 'h', 'a', 'o', '*', '#', 'M', 'W', '&', '8', '%', 'B',
        '@', '$',
    ];

    // 3. Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let mut ramp_size = 16usize;
    let mut raster_size = 32.0f32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ramp-size" => {
                if i + 1 < args.len() {
                    ramp_size = args[i + 1].parse().expect("Invalid --ramp-size");
                    i += 2;
                } else {
                    eprintln!("--ramp-size requires a value");
                    std::process::exit(1);
                }
            }
            "--font-size" => {
                if i + 1 < args.len() {
                    raster_size = args[i + 1].parse().expect("Invalid --font-size");
                    i += 2;
                } else {
                    eprintln!("--font-size requires a value");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    // 4. Measure ink density for each candidate
    // fontdue::rasterize returns (metrics, bitmap) where bitmap is u8 per-pixel coverage [0,255]
    let mut densities: Vec<(char, f32)> = Vec::with_capacity(candidates.len());
    for &ch in candidates {
        let (_, bitmap) = font.rasterize(ch, raster_size);
        let density = if bitmap.is_empty() {
            0.0
        } else {
            let sum: u64 = bitmap.iter().map(|&v| v as u64).sum();
            sum as f32 / (255.0 * bitmap.len() as f32)
        };
        densities.push((ch, density));
    }

    // 5. Sort by density ascending (stable sort for determinism)
    densities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // 6. Pick N target densities evenly spaced between min and max observed
    let min_d = densities.first().unwrap().1;
    let max_d = densities.last().unwrap().1;
    let targets: Vec<f32> = (0..ramp_size)
        .map(|i| min_d + (max_d - min_d) * (i as f32 / (ramp_size - 1) as f32))
        .collect();

    // 7. For each target, greedily pick the closest unused candidate
    let mut used = vec![false; densities.len()];
    let mut ramp_chars = Vec::with_capacity(ramp_size);
    for target in targets {
        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        for (idx, &(_, d)) in densities.iter().enumerate() {
            if used[idx] {
                continue;
            }
            let dist = (d - target).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        used[best_idx] = true;
        ramp_chars.push(densities[best_idx].0);
    }

    // 8. Emit the Rust literal for ramp.rs
    print!("pub const RAMP: &[char] = &[");
    for (i, ch) in ramp_chars.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        // Escape for Rust char literal
        match ch {
            '\'' => print!("'\\''"),
            '\\' => print!("'\\\\'"),
            '\n' => print!("'\\n'"),
            '\r' => print!("'\\r'"),
            '\t' => print!("'\\t'"),
            '\0' => print!("'\\0'"),
            c if c.is_ascii_control() => print!("'\\x{:02x}'", *c as u8),
            c => print!("'{}'", c),
        }
    }
    println!("];");

    // Also emit a debug summary to stderr (not captured by stdout -> ramp.rs copy)
    eprintln!("Ramp generated: {} chars", ramp_chars.len());
    eprintln!("Density range: [{:.4}, {:.4}]", min_d, max_d);
    for (i, (ch, d)) in densities.iter().enumerate() {
        let used_mark = if used[i] { " *" } else { "" };
        eprintln!("  {:2}: '{}' density={:.4}{}", i, ch, d, used_mark);
    }
}