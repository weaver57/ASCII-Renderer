use ascii_renderer::pool::{BufferPool, PoolGuard};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn test_buffer_pool_exhaustion_and_blocking() {
    const POOL_SIZE: usize = 3;
    let pool = Arc::new(BufferPool::new(POOL_SIZE, || vec![0u8; 1024]));

    // Acquire all N buffers
    let mut guards = Vec::new();
    for i in 0..POOL_SIZE {
        let mut g = pool.acquire();
        g[0] = (i + 1) as u8;
        guards.push(g);
    }
    assert_eq!(pool.free_count(), 0);

    // Spawn thread that attempts to acquire the (N+1)th buffer
    let pool_clone = Arc::clone(&pool);
    let acquired_flag = Arc::new(AtomicBool::new(false));
    let flag_clone = Arc::clone(&acquired_flag);

    let handle = thread::spawn(move || {
        let mut g = pool_clone.acquire();
        assert_eq!(g[0], 1); // Should receive the first released buffer
        g[0] = 99;
        flag_clone.store(true, Ordering::SeqCst);
    });

    // Verify thread is blocked
    thread::sleep(Duration::from_millis(60));
    assert!(
        !acquired_flag.load(Ordering::SeqCst),
        "Thread should remain blocked while pool is exhausted"
    );

    // Release one buffer
    let first = guards.remove(0);
    drop(first);

    // The thread should unblock promptly
    let start = Instant::now();
    handle.join().expect("Worker thread joined successfully");
    assert!(acquired_flag.load(Ordering::SeqCst));
    assert!(start.elapsed() < Duration::from_millis(200));

    // Release the rest
    drop(guards);
    assert_eq!(pool.free_count(), POOL_SIZE);
}

#[test]
fn test_buffer_pool_raii_panic_safety() {
    let pool = Arc::new(BufferPool::new(2, || vec![123u32; 16]));
    assert_eq!(pool.free_count(), 2);

    let pool_clone = Arc::clone(&pool);
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let mut g = pool_clone.acquire();
        g[0] = 777;
        assert_eq!(pool_clone.free_count(), 1);
        panic!("Simulated worker panic");
    }));

    assert!(panic_result.is_err(), "Expected panic was caught");
    // Pool must have all 2 buffers back safely despite the unwinding panic
    assert_eq!(
        pool.free_count(),
        2,
        "Buffer must be returned to pool after panic unwind"
    );
}

#[test]
fn test_buffer_pool_concurrent_stress() {
    const POOL_CAP: usize = 4;
    const NUM_THREADS: usize = 8;
    const ITERATIONS_PER_THREAD: usize = 500;

    let pool = Arc::new(BufferPool::new(POOL_CAP, || vec![0u32; 256]));
    let total_operations = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for thread_id in 0..NUM_THREADS {
        let pool_clone = Arc::clone(&pool);
        let ops_clone = Arc::clone(&total_operations);

        handles.push(thread::spawn(move || {
            for i in 0..ITERATIONS_PER_THREAD {
                let mut guard = pool_clone.acquire();
                guard[0] = (thread_id * 1000 + i) as u32;
                // Brief artificial work
                let val = guard[0];
                assert_eq!(val, (thread_id * 1000 + i) as u32);
                ops_clone.fetch_add(1, Ordering::Relaxed);
                // Guard dropped at end of iteration
            }
        }));
    }

    for h in handles {
        h.join().expect("Stress worker finished");
    }

    assert_eq!(
        total_operations.load(Ordering::Relaxed),
        NUM_THREADS * ITERATIONS_PER_THREAD
    );
    assert_eq!(
        pool.free_count(),
        POOL_CAP,
        "All pool slots must be returned after stress completion"
    );
}

#[test]
fn test_into_inner_and_wrap_lifecycle() {
    let pool = BufferPool::new(1, || vec![10u8, 20, 30]);

    // Acquire, extract inner vector
    let guard = pool.acquire();
    let free_tx = pool.free_sender();
    let mut raw = guard.into_inner();
    raw.push(40);

    // Pool remains empty while raw is held outside
    assert_eq!(pool.free_count(), 0);

    // Wrap raw in a new guard and drop it
    let new_guard = PoolGuard::wrap(raw, free_tx);
    drop(new_guard);

    // Pool now has the modified buffer back
    assert_eq!(pool.free_count(), 1);
    let g = pool.acquire();
    assert_eq!(&*g, &[10u8, 20, 30, 40]);
}
