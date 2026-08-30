use std::fs;
use std::path::PathBuf;

/// Returns the path to the config file: `~/.ascii_renderer.toml`
fn config_path() -> Option<PathBuf> {
    dirs_home().map(|h| h.join(".ascii_renderer.toml"))
}

/// Get the home directory in a cross-platform way (no crate dependency).
fn dirs_home() -> Option<PathBuf> {
    if let Ok(profile) = std::env::var("USERPROFILE") {
        return Some(PathBuf::from(profile));
    }
    if let Ok(home) = std::env::var("HOME") {
        return Some(PathBuf::from(home));
    }
    None
}

/// Parsed config values.
pub struct Config {
    pub char_aspect: Option<f32>,
}

impl Config {
    /// Load config from disk. Returns defaults if the file doesn't exist or is
    /// malformed.
    pub fn load() -> Self {
        let path = match config_path() {
            Some(p) => p,
            None => return Self { char_aspect: None },
        };

        let contents = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => return Self { char_aspect: None },
        };

        Self::parse(&contents)
    }

    /// Parse config from a string (used by both load() and tests).
    pub fn parse(contents: &str) -> Self {
        let mut char_aspect = None;
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Parse "key = value" lines
            if let Some((key, value)) = line.split_once('=') {
                if key.trim() == "char_aspect" {
                    if let Ok(v) = value.trim().parse::<f32>() {
                        if v > 0.0 && v.is_finite() && v >= 0.1 && v <= 1.0 {
                            char_aspect = Some(v);
                        }
                    }
                }
            }
        }

        Self { char_aspect }
    }

    /// Save a char_aspect value to the config file, creating or overwriting it.
    /// Preserves any other lines in the file.
    pub fn save_char_aspect(value: f32) -> anyhow::Result<()> {
        let path = config_path().ok_or_else(|| {
            anyhow::anyhow!("Could not determine home directory for config file")
        })?;

        let mut lines: Vec<String> = fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .map(|l| l.to_string())
            .collect();

        let new_line = format!("char_aspect = {}", value);

        let mut found = false;
        for line in &mut lines {
            let trimmed = line.trim();
            if trimmed.starts_with("char_aspect") && trimmed.contains('=') {
                *line = new_line.clone();
                found = true;
                break;
            }
        }
        if !found {
            lines.push(new_line);
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut content = lines.join("\n");
        if !content.ends_with('\n') {
            content.push('\n');
        }

        fs::write(&path, &content)?;
        eprintln!("[ascii_renderer] Saved char_aspect = {} to {:?}", value, path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parse_no_char_aspect() {
        let config = Config::parse("");
        assert!(config.char_aspect.is_none());
    }

    #[test]
    fn config_parse_garbage_ignored() {
        let config = Config::parse("random text
no_key
");
        assert!(config.char_aspect.is_none());
    }

    #[test]
    fn config_parse_valid() {
        let config = Config::parse("char_aspect = 0.45");
        assert_eq!(config.char_aspect, Some(0.45));
    }

    #[test]
    fn config_parse_with_comments() {
        let config = Config::parse("# My config\nchar_aspect = 0.55\n# other setting");
        assert_eq!(config.char_aspect, Some(0.55));
    }

    #[test]
    fn config_parse_empty() {
        let config = Config::parse("");
        assert!(config.char_aspect.is_none());
    }

    #[test]
    fn config_parse_invalid_value() {
        let config = Config::parse("char_aspect = abc");
        assert!(config.char_aspect.is_none());
    }

    #[test]
    fn config_parse_out_of_range() {
        let config = Config::parse("char_aspect = 2.0");
        assert!(config.char_aspect.is_none());
    }
}
