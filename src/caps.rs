//! Terminal capability detection: how many colors the terminal can display, and
//! whether it supports synchronized output (mode 2026).
//!
//! Color depth is detected via the `supports-color` crate (§4.1), which
//! handles the long tail of `COLORTERM`/`TERM`/`NO_COLOR`/CI platform quirks
//! far more robustly than hand-rolling the parsing would.
//!
//! Synchronized-output support is probed with a DECRQM round trip (`\x1b[?2026$p`
//! → `\x1b[?2026;<status>$y`) — but on Windows this query is skipped entirely
//! (D7): raw escape-sequence round trips behave differently and less reliably
//! across Windows console hosts, so we rely on static heuristics there.

// `Read`/`Write` traits are only exercised by the DECRQM round trip, which is
// compiled out on non-Unix (D7), so gate the import to avoid a Windows warning.
#[cfg(unix)]
#[cfg(unix)]
use std::io::{Read, Write};
// `Duration` is used by the always-compiled `SYNC_QUERY_TIMEOUT` constant;
// `Instant` is only used by the DECRQM round trip, compiled out on non-Unix.
use std::time::Duration;
#[cfg(unix)]
use std::time::Instant;

/// Terminal color capability, resolved once at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSupport {
    TrueColor,
    Palette256,
    Basic16,
    NoColor,
}

/// Detect the terminal's color depth — run once at startup, not per frame.
pub fn detect_color_support() -> ColorSupport {
    match supports_color::on(supports_color::Stream::Stdout) {
        None => ColorSupport::NoColor,
        Some(level) if level.has_16m => ColorSupport::TrueColor,
        Some(level) if level.has_256 => ColorSupport::Palette256,
        Some(_) => ColorSupport::Basic16,
    }
}

/// How long to wait for a DECRQM reply. A terminal that supports mode 2026
/// replies near-instantly; one that doesn't may not reply at all, so there is
/// no reason to wait long — failing closed keeps startup snappy on terminals
/// that simply ignore the query.
const SYNC_QUERY_TIMEOUT: Duration = Duration::from_millis(100);

/// The DECRQM query for private mode 2026 (synchronized output).
const SYNC_QUERY: &[u8] = b"\x1b[?2026$p";
/// DECRQM reply prefix for mode 2026: `CSI ? 2026 ;`.
const SYNC_REPLY_MARKER: &[u8] = b"\x1b[?2026;";

/// Attempt to detect synchronized-output support via a DECRQM round trip.
///
/// Returns `true` only if the terminal definitively reports it recognizes mode
/// 2026 (DECRQM status 1, 2, or 3). Everything else — no reply within the
/// timeout, status 0 or 4 (GNOME Terminal reports 4, functionally
/// "permanently disabled"), a malformed reply, or an I/O error — fails closed
/// to `false`. On Windows the query is skipped outright (D7).
pub fn query_sync_output_support() -> bool {
    #[cfg(unix)]
    {
        query_sync_output_support_unix()
    }
    #[cfg(not(unix))]
    {
        // Windows: skip the escape-sequence round trip entirely (D7). Both
        // constants are only otherwise used by the #[cfg(unix)] helper, so
        // reference them here to keep the Windows build free of unused-const
        // warnings.
        let _ = (SYNC_QUERY, SYNC_QUERY_TIMEOUT);
        false
    }
}

#[cfg(unix)]
fn query_sync_output_support_unix() -> bool {
    // Write the query.
    let mut stdout = std::io::stdout();
    if stdout.write_all(SYNC_QUERY).is_err() {
        return false;
    }
    if stdout.flush().is_err() {
        return false;
    }

    let stdin = std::io::stdin();
    let fd = libc::STDIN_FILENO;
    let deadline = Instant::now() + SYNC_QUERY_TIMEOUT;
    let mut acc: Vec<u8> = Vec::new();

    loop {
        // Check whether the accumulated bytes already form a complete reply.
        if let Some(supported) = parse_sync_reply(&acc) {
            return supported;
        }

        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let remaining_ms = deadline.saturating_duration_since(Instant::now()).as_millis();
        if remaining_ms == 0 {
            // Timed out with no complete reply → unsupported (fail closed).
            return false;
        }
        let remaining_ms = remaining_ms.min(i32::MAX as u128) as i32;

        // Borrow stdin's underlying fd without reading through it (we need raw
        // byte reads; a poll-then-read on the fd is non-blocking in practice).
        let _ = &stdin;
        let rc = unsafe { libc::poll(&mut pfd, 1, remaining_ms) };
        if rc <= 0 {
            // rc == 0 → timeout; rc < 0 → poll error. Both fail closed.
            return false;
        }

        // Poll signalled readability; read what's available (short reads on a
        // tty are fine here — we keep polling until the reply is complete or
        // the deadline lapses).
        let mut buf = [0u8; 64];
        let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        if n <= 0 {
            // read error or EOF before a complete reply → fail closed.
            return false;
        }
        acc.extend_from_slice(&buf[..n as usize]);
    }
}

/// Parses a (possibly partial / multi-chunk) DECRQM reply accumulator for mode
/// 2026. Returns `Some(true)` for status 1/2/3 (supported), `Some(false)` for
/// status 0/4 or any malformed/negative status (not supported), and `None`
/// while the reply is still incomplete (no full match yet).
pub fn parse_sync_reply(acc: &[u8]) -> Option<bool> {
    let pos = find_subslice(acc, SYNC_REPLY_MARKER)?;
    let rest = &acc[pos + SYNC_REPLY_MARKER.len()..];

    // Collect the status digits, then expect `$y`.
    let mut status_digits = String::new();
    let mut i = 0usize;
    while i < rest.len() {
        let b = rest[i];
        if b.is_ascii_digit() {
            status_digits.push(b as char);
            i += 1;
        } else if b == b'$' {
            // The reply terminator is `$y`. If `y` isn't here yet, the reply is
            // still pending (the terminator byte may arrive in a later chunk);
            // once the byte after `$` is present it must be `y`, else malformed.
            return match rest.get(i + 1) {
                Some(b'y') => Some(status_is_supported(&status_digits)),
                Some(_) => Some(false), // `$` followed by a non-`y` byte.
                None => None,           // `$` at the end → `y` may still arrive.
            };
        } else {
            // Unexpected byte before the terminator → not a valid reply.
            return Some(false);
        }
    }
    // Ran off the end of the accumulator → reply still pending.
    None
}

fn status_is_supported(digits: &str) -> bool {
    match digits {
        "1" | "2" | "3" => true, // set / reset / permanently set
        "0" | "4" => false,      // not recognized / permanently disabled
        _ => false,              // other status values → fail closed
    }
}

/// Byte-subslice search (no allocator-free memmem in std for byte slices).
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_supported_statuses() {
        assert_eq!(parse_sync_reply(b"\x1b[?2026;1$y"), Some(true));
        assert_eq!(parse_sync_reply(b"\x1b[?2026;2$y"), Some(true));
        assert_eq!(parse_sync_reply(b"\x1b[?2026;3$y"), Some(true));
    }

    #[test]
    fn parse_unsupported_statuses() {
        assert_eq!(parse_sync_reply(b"\x1b[?2026;0$y"), Some(false));
        assert_eq!(parse_sync_reply(b"\x1b[?2026;4$y"), Some(false));
        assert_eq!(parse_sync_reply(b"\x1b[?2026;9$y"), Some(false));
    }

    #[test]
    fn parse_partial_returns_none() {
        assert_eq!(parse_sync_reply(b""), None);
        assert_eq!(parse_sync_reply(b"\x1b"), None);
        assert_eq!(parse_sync_reply(b"\x1b[?"), None);
        assert_eq!(parse_sync_reply(b"\x1b[?2026;"), None);
        assert_eq!(parse_sync_reply(b"\x1b[?2026;1"), None);
        assert_eq!(parse_sync_reply(b"\x1b[?2026;1$"), None);
    }

    #[test]
    fn parse_reply_embedded_in_noise() {
        // A reply arriving after unrelated prior bytes must still be found.
        assert_eq!(parse_sync_reply(b"junk\x1b[?2026;2$y"), Some(true));
    }

    #[test]
    fn parse_split_across_chunks() {
        // Bytes split across multiple reads must still assemble correctly.
        let mut acc = Vec::new();
        acc.extend_from_slice(b"\x1b[?20");
        assert_eq!(parse_sync_reply(&acc), None);
        acc.extend_from_slice(b"26;1");
        assert_eq!(parse_sync_reply(&acc), None);
        acc.extend_from_slice(b"$y");
        assert_eq!(parse_sync_reply(&acc), Some(true));
    }
}
