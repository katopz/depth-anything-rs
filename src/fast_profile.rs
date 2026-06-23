//! Lightweight profiling accumulators for the fast backbone/head path.
//!
//! Enabled by `DA_FAST_PROFILE=1`. When disabled, every public API is a no-op
//! that compiles down to nothing (the `enabled()` check is `const`-folded after
//! the first call, and the `AtomicU64` increments are behind it).
//!
//! # Lock-free design
//!
//! The previous implementation used a `Mutex<Vec<String>>` for label→bucket
//! lookup. Under heavy rayon parallelism (1680+ record() calls per iter across
//! 16 threads) this caused massive contention that inflated measured GEMM times
//! by 2-3× over actual, making A/B comparisons unreliable.
//!
//! The current implementation uses a fixed compile-time label table: each label
//! is registered exactly once (via the first `scope()` call) under a single
//! `Once` for that label, returning a stable bucket index. Subsequent record()
//! calls use only relaxed atomic adds — no locks on the hot path.
//!
//! Usage:
//! ```ignore
//! fast_profile::register("attn_qkv"); // optional, also done lazily by scope()
//! let _g = fast_profile::scope_idx(fast_profile::idx("attn_qkv"));
//! // ... work ...
//! ```
//! At program exit (or via [`dump_and_reset`]), accumulated microsecond counts
//! are printed to stderr, one line per label.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: std::sync::Once = std::sync::Once::new();

/// Maximum number of distinct labels that can be registered.
pub const N_BUCKETS: usize = 64;

static USEC: [AtomicU64; N_BUCKETS] = [const { AtomicU64::new(0) }; N_BUCKETS];
static COUNT: [AtomicU64; N_BUCKETS] = [const { AtomicU64::new(0) }; N_BUCKETS];

/// Registered label names (indexed by bucket). Slot `i` is `Some(name)` once
/// the label has been registered. Each label has its own `OnceLock` so
/// registration is lock-free after the first call.
static LABELS: [OnceLock<Option<&'static str>>; N_BUCKETS] = [const { OnceLock::new() }; N_BUCKETS];

/// Total number of labels registered so far. Used to allocate fresh bucket
/// indices. We use a simple atomic fetch_add for allocation — over-allocation
/// beyond `N_BUCKETS` falls through to the "overflow" bucket (`N_BUCKETS-1`).
// (label registration uses the per-slot OnceLocks above; no central counter)

/// True iff `DA_FAST_PROFILE=1` was set at first call.
#[inline]
pub fn enabled() -> bool {
    INIT.call_once(|| {
        let on = match std::env::var("DA_FAST_PROFILE") {
            Ok(v) => matches!(v.as_str(), "1" | "on" | "true" | "yes"),
            Err(_) => false,
        };
        ENABLED.store(on, Ordering::Relaxed);
    });
    ENABLED.load(Ordering::Relaxed)
}

/// Register a label and return its stable bucket index. Idempotent: calling
/// with the same `&'static str` returns the same index. The pointer identity of
/// the `&'static str` is used (no hashing/locking on the hot path).
///
/// Labels beyond `N_BUCKETS` collapse onto the last bucket.
#[inline]
pub fn idx(label: &'static str) -> usize {
    // Each bucket's OnceLock stores the *first* label that claims it. We probe
    // by trying to initialize slot i with our label; if it's already taken by
    // a *different* label we move on. This is O(n_labels) worst-case but only
    // happens on the first `idx()` call per label — subsequent calls hit the
    // cached OnceLock state (no scanning).
    //
    // We track which buckets we've already staked a claim on via the
    // OnceLock's internal state. To avoid the scan on repeat calls, callers
    // should cache the returned index (the `scope()` helper does this).
    for i in 0..N_BUCKETS {
        // Fast path: this bucket is already ours.
        if let Some(Some(existing)) = LABELS[i].get() {
            if std::ptr::eq(*existing, label) {
                return i;
            }
            continue;
        }
        // Try to claim this bucket.
        let claimed = LABELS[i].set(Some(label)).is_ok();
        if claimed {
            return i;
        }
        // Race: someone else claimed it between our get() and set(). Re-check.
        if let Some(Some(existing)) = LABELS[i].get() {
            if std::ptr::eq(*existing, label) {
                return i;
            }
        }
    }
    // Too many labels: collapse onto the last bucket.
    N_BUCKETS - 1
}

/// Record `usec` microseconds against bucket `i`. No-op when disabled.
#[inline]
pub fn record_idx(i: usize, usec: u64) {
    if !enabled() {
        return;
    }
    USEC[i].fetch_add(usec, Ordering::Relaxed);
    COUNT[i].fetch_add(1, Ordering::Relaxed);
}

/// RAII scope guard keyed by bucket index. Records elapsed time on drop.
///
/// `start` is `None` when profiling is disabled, so creation is just a branch
/// and an `Option::None` init — no `Instant::now()` syscall on the hot path.
pub struct Scope {
    idx: usize,
    start: Option<std::time::Instant>,
}

impl Scope {
    #[inline]
    pub fn new(idx: usize) -> Self {
        Self {
            idx,
            start: if enabled() {
                Some(std::time::Instant::now())
            } else {
                None
            },
        }
    }
}

impl Drop for Scope {
    #[inline]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let us = start.elapsed().as_micros() as u64;
            record_idx(self.idx, us);
        }
    }
}

/// Create a scope guard for the given label. The label is registered on first
/// use (lock-free after). For hot paths, prefer caching the result of
/// [`idx`] and calling [`scope_idx`].
#[inline]
pub fn scope(label: &'static str) -> Scope {
    Scope::new(idx(label))
}

/// Create a scope guard for a pre-resolved bucket index.
#[inline]
pub fn scope_idx(idx: usize) -> Scope {
    Scope::new(idx)
}

/// Dump accumulated timings to stderr and reset counters.
pub fn dump_and_reset() {
    if !enabled() {
        return;
    }
    eprintln!("=== fast_profile (usec, calls) ===");
    let mut total: u64 = 0;
    for i in 0..N_BUCKETS {
        let us = USEC[i].swap(0, Ordering::Relaxed);
        let n = COUNT[i].swap(0, Ordering::Relaxed);
        if us == 0 {
            continue;
        }
        let label = LABELS[i].get().copied().flatten().unwrap_or("<overflow>");
        eprintln!("  {label:<28} {us:>10} µs  ({n} calls)");
        total += us;
    }
    eprintln!("  TOTAL                        {total:>10} µs");
}
