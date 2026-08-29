use std::io::{self, Stdout, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, size, Clear, ClearType,
        EnterAlternateScreen, LeaveAlternateScreen,
    },
};

static PANIC_HOOK_SET: AtomicBool = AtomicBool::new(false);

/// RAII guard to ensure the terminal is restored to its original state even on panic or early exit.
pub struct TerminalGuard {
    stdout: Stdout,
    active: Arc<AtomicBool>,
}

impl TerminalGuard {
    pub fn init() -> io::Result<Self> {
        let active = Arc::new(AtomicBool::new(true));

        if !PANIC_HOOK_SET.swap(true, Ordering::SeqCst) {
            let default_panic_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |panic_info| {
                let mut stdout = io::stdout();
                let _ = execute!(stdout, Show, LeaveAlternateScreen);
                let _ = disable_raw_mode();
                let _ = stdout.flush();
                default_panic_hook(panic_info);
            }));
        }

        let mut stdout = io::stdout();
        enable_raw_mode()?;
        execute!(
            stdout,
            EnterAlternateScreen,
            Hide,
            Clear(ClearType::All),
            MoveTo(0, 0)
        )?;
        stdout.flush()?;

        Ok(Self { stdout, active })
    }

    /// Returns the current terminal dimensions as (columns, rows).
    pub fn get_size() -> io::Result<(u16, u16)> {
        size()
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.active.store(false, Ordering::SeqCst);
        let _ = execute!(
            self.stdout,
            Show,
            LeaveAlternateScreen
        );
        let _ = disable_raw_mode();
        let _ = self.stdout.flush();
    }
}
