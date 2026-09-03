//! Generic cross-thread buffer pool with RAII guard lifecycle (`BufferPool<T>` and `PoolGuard<T>`).
//!
//! Provides pre-allocated, bounded buffer recycling across pipeline threads (Design Decision D5).
//! Any buffer acquired via `acquire()` is wrapped in a `PoolGuard<T>` whose `Drop` implementation
//! automatically returns the buffer back to the pool on normal completion, early returns, errors,
//! or unwinding panics.
//!
//! Calling `into_inner()` defuses the guard and extracts the inner value when ownership
//! transfers across a channel boundary.

use std::ops::{Deref, DerefMut};

/// Generic pool of pre-allocated buffers backed by a bounded crossbeam channel.
pub struct BufferPool<T> {
    free_tx: crossbeam_channel::Sender<T>,
    free_rx: crossbeam_channel::Receiver<T>,
    capacity: usize,
}

impl<T> BufferPool<T> {
    /// Pre-seeds the pool with `count` instances constructed via `make`.
    pub fn new(count: usize, make: impl Fn() -> T) -> Self {
        assert!(count > 0, "BufferPool capacity must be greater than zero");
        let (tx, rx) = crossbeam_channel::bounded(count);
        for _ in 0..count {
            tx.send(make())
                .expect("BufferPool initialization failed: capacity matches count");
        }
        BufferPool {
            free_tx: tx,
            free_rx: rx,
            capacity: count,
        }
    }

    /// Blocks until a buffer is available from the pool and returns a `PoolGuard<T>`.
    pub fn acquire(&self) -> PoolGuard<T> {
        let item = self
            .free_rx
            .recv()
            .expect("BufferPool free_rx disconnected while pool is alive");
        PoolGuard {
            item: Some(item),
            return_to: self.free_tx.clone(),
        }
    }

    /// Attempts to acquire a buffer without blocking. Returns `None` if the pool is empty.
    pub fn try_acquire(&self) -> Option<PoolGuard<T>> {
        match self.free_rx.try_recv() {
            Ok(item) => Some(PoolGuard {
                item: Some(item),
                return_to: self.free_tx.clone(),
            }),
            Err(_) => None,
        }
    }

    /// Returns a clone of the pool's return channel sender, allowing guards to be constructed
    /// around buffers received from external channels.
    pub fn free_sender(&self) -> crossbeam_channel::Sender<T> {
        self.free_tx.clone()
    }

    /// Returns the number of currently available (idle) buffers in the pool.
    pub fn free_count(&self) -> usize {
        self.free_rx.len()
    }

    /// Returns the total capacity of the pool.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// RAII guard wrapping an acquired buffer.
///
/// When dropped, returns the wrapped buffer back to the originating `BufferPool<T>`
/// unless ownership was explicitly extracted via `into_inner()`.
pub struct PoolGuard<T> {
    item: Option<T>,
    return_to: crossbeam_channel::Sender<T>,
}

impl<T> PoolGuard<T> {
    /// Wraps an existing buffer instance with a return sender.
    pub fn wrap(item: T, return_to: crossbeam_channel::Sender<T>) -> Self {
        PoolGuard {
            item: Some(item),
            return_to,
        }
    }

    /// Extracts the inner value and defuses the guard's `Drop` return.
    ///
    /// Use this when transferring ownership across a channel boundary.
    pub fn into_inner(mut self) -> T {
        self.item
            .take()
            .expect("PoolGuard item was already taken or defused")
    }

    /// Returns `true` if the guard currently holds a valid buffer.
    pub fn is_active(&self) -> bool {
        self.item.is_some()
    }
}

impl<T> Deref for PoolGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item
            .as_ref()
            .expect("Cannot dereference a defused PoolGuard")
    }
}

impl<T> DerefMut for PoolGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item
            .as_mut()
            .expect("Cannot dereference a defused PoolGuard")
    }
}

impl<T> Drop for PoolGuard<T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            // Best-effort send back to the pool. If the pool receiver has been dropped
            // during process shutdown, ignore the error cleanly.
            let _ = self.return_to.send(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_pool_creation_and_capacity() {
        let pool = BufferPool::new(4, || vec![0u8; 1024]);
        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn test_acquire_and_auto_return_on_drop() {
        let pool = BufferPool::new(2, || vec![0u8; 100]);
        assert_eq!(pool.free_count(), 2);

        {
            let mut guard1 = pool.acquire();
            assert_eq!(pool.free_count(), 1);
            guard1[0] = 42;

            {
                let mut guard2 = pool.acquire();
                assert_eq!(pool.free_count(), 0);
                guard2[0] = 99;
                assert_eq!(guard2[0], 99);
            }
            // guard2 dropped -> returned
            assert_eq!(pool.free_count(), 1);
            assert_eq!(guard1[0], 42);
        }
        // guard1 dropped -> returned
        assert_eq!(pool.free_count(), 2);
    }

    #[test]
    fn test_try_acquire() {
        let pool = BufferPool::new(1, || String::from("hello"));
        let g1 = pool.try_acquire();
        assert!(g1.is_some());
        assert_eq!(pool.free_count(), 0);

        let g2 = pool.try_acquire();
        assert!(g2.is_none());

        drop(g1);
        assert_eq!(pool.free_count(), 1);
        let g3 = pool.try_acquire();
        assert!(g3.is_some());
    }

    #[test]
    fn test_into_inner_defuses_drop() {
        let pool = BufferPool::new(2, || vec![1, 2, 3]);
        assert_eq!(pool.free_count(), 2);

        let guard = pool.acquire();
        assert_eq!(pool.free_count(), 1);

        let raw_vec = guard.into_inner();
        assert_eq!(raw_vec, vec![1, 2, 3]);
        // Guard was defused, so free_count remains 1
        assert_eq!(pool.free_count(), 1);

        // We can wrap the raw vec back into a guard and drop it to return to the pool
        let guard_wrapped = PoolGuard::wrap(raw_vec, pool.free_sender());
        drop(guard_wrapped);
        assert_eq!(pool.free_count(), 2);
    }

    #[test]
    fn test_acquire_blocks_until_released() {
        let pool = Arc::new(BufferPool::new(1, || vec![0u8; 10]));
        let guard = pool.acquire();
        assert_eq!(pool.free_count(), 0);

        let acquired_flag = Arc::new(AtomicBool::new(false));
        let flag_clone = Arc::clone(&acquired_flag);
        let pool_clone = Arc::clone(&pool);

        let handle = thread::spawn(move || {
            let _g = pool_clone.acquire();
            flag_clone.store(true, Ordering::SeqCst);
        });

        // Ensure background thread is blocked
        thread::sleep(Duration::from_millis(50));
        assert!(!acquired_flag.load(Ordering::SeqCst));

        // Release the buffer
        drop(guard);

        handle.join().unwrap();
        assert!(acquired_flag.load(Ordering::SeqCst));
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn test_raii_return_on_panic_unwind() {
        let pool = Arc::new(BufferPool::new(1, || vec![100u8]));
        assert_eq!(pool.free_count(), 1);

        let pool_clone = Arc::clone(&pool);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let mut _guard = pool_clone.acquire();
            assert_eq!(pool_clone.free_count(), 0);
            panic!("Intentional panic to test RAII unwind safety");
        }));

        assert!(result.is_err());
        // Despite the panic, RAII Drop must have returned the buffer to the pool!
        assert_eq!(pool.free_count(), 1);
    }
}
