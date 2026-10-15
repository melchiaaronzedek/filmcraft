//! How much memory this process's caches and decoders may use.
//!
//! Every cache used to have a fixed size picked for one workload: 64 decoded frames per clip, a
//! 1 GB plan cache, 24 GPU uploads, frame-threaded decoders with `threads + 2` pictures in flight.
//! At 4K 10-bit 4:2:2 (33 MB a frame) two clips added up to more than a 16 GB laptop has. Here
//! the budgets come from the machine instead: its RAM, minus what Settings ▸ Memory ▸ "RAM
//! reserved for other applications" keeps free ([`set_reserved_bytes`]), split between the
//! caches ([`Budget`]), and halved while the machine is short of memory ([`under_pressure`]).
//!
//! RAM is read from `/proc/meminfo` on Linux (pure Rust, no FFI). Elsewhere it is unknown until a
//! host reports it ([`set_system_reporter`]); budgets then assume a machine with [`FALLBACK_TOTAL`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// RAM of the machine and the part of it available to start new work without swapping (bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemMemory {
    pub total: u64,
    pub available: u64,
}

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

/// Total RAM assumed where it cannot be read (and no host reports it).
pub const FALLBACK_TOTAL: u64 = 16 * GIB;

/// The app's budget never goes below this, whatever is reserved: below it playback cannot work.
const MIN_APP_BUDGET: u64 = 3 * GIB / 2;

/// Default of Settings ▸ Memory ▸ "RAM reserved for other applications".
pub const DEFAULT_RESERVED: u64 = 4 * GIB;

static RESERVED: AtomicU64 = AtomicU64::new(DEFAULT_RESERVED);

/// How long a reading of the system's memory is reused: callers ask per decoded frame.
const REFRESH: Duration = Duration::from_millis(250);

/// Settings ▸ Memory ▸ "RAM reserved for other applications" (bytes): the caches size themselves
/// so the process leaves this much of the machine's RAM to other programs.
pub fn set_reserved_bytes(bytes: u64) {
    RESERVED.store(bytes, Ordering::Relaxed);
}

pub fn reserved_bytes() -> u64 {
    RESERVED.load(Ordering::Relaxed)
}

/// A host's way to read the machine's memory where `/proc/meminfo` does not exist (macOS,
/// Windows: `crates/platform` registers one at start-up).
pub type SystemReporter = fn() -> Option<SystemMemory>;

static REPORTER: OnceLock<SystemReporter> = OnceLock::new();

/// Register the host's memory reader (the first call wins).
pub fn set_system_reporter(f: SystemReporter) {
    let _ = REPORTER.set(f);
}

/// MemTotal and MemAvailable from the text of `/proc/meminfo` (kB values).
pub fn parse_meminfo(text: &str) -> Option<SystemMemory> {
    let field = |name: &str| -> Option<u64> {
        let line = text.lines().find(|l| l.starts_with(name))?;
        let kb: u64 = line.get(name.len()..)?.split_whitespace().next()?.parse().ok()?;
        kb.checked_mul(1024)
    };
    let total = field("MemTotal:")?;
    // kernels before 3.14 have no MemAvailable: free + page cache is the old approximation
    let available = field("MemAvailable:").or_else(|| Some(field("MemFree:")?.saturating_add(field("Cached:").unwrap_or(0))))?;
    (total > 0).then_some(SystemMemory { total, available: available.min(total) })
}

fn read_system() -> Option<SystemMemory> {
    if let Some(f) = REPORTER.get() {
        return f();
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo").ok().as_deref().and_then(parse_meminfo)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// The machine's memory now (read at most every [`REFRESH`]); `None` where it cannot be read.
pub fn system() -> Option<SystemMemory> {
    static LAST: Mutex<Option<(Instant, Option<SystemMemory>)>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(PoisonError::into_inner);
    match *last {
        Some((at, m)) if at.elapsed() < REFRESH => m,
        _ => {
            let m = read_system();
            *last = Some((Instant::now(), m));
            m
        }
    }
}

/// Bytes this process's caches and decoders may use together: the machine's RAM minus the
/// reserve (at least [`MIN_APP_BUDGET`]).
pub fn app_budget() -> u64 {
    app_budget_for(system().map_or(FALLBACK_TOTAL, |m| m.total), reserved_bytes())
}

fn app_budget_for(total: u64, reserved: u64) -> u64 {
    total.saturating_sub(reserved).max(MIN_APP_BUDGET.min(total.saturating_mul(3) / 4))
}

/// The machine is short of memory: less than a tenth of its RAM (at least 1 GiB) is available.
/// Caches then keep half their budget, idle ones nothing, and decoders fewer pictures in flight.
pub fn under_pressure() -> bool {
    system().is_some_and(pressure_for)
}

fn pressure_for(m: SystemMemory) -> bool {
    m.available < (m.total / 10).max(GIB)
}

/// A share of the app's memory: a percentage of [`app_budget`] within `[min, max]`, halved under
/// pressure (but never below `min`).
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub percent: u64,
    pub min: u64,
    pub max: u64,
}

impl Budget {
    /// Decoded frames held by all the clips' frame caches together.
    pub const DECODED_FRAMES: Budget = Budget { percent: 12, min: 384 * MIB, max: 3 * GIB };
    /// The pictures one decoder works on in parallel (frame threading).
    pub const DECODER: Budget = Budget { percent: 4, min: 192 * MIB, max: 640 * MIB };
    /// Prepared compositing plans of the monitors (GPU path).
    pub const PLANS: Budget = Budget { percent: 6, min: 128 * MIB, max: GIB };
    /// Finished RGBA frames of the monitors, thumbnails and scopes (CPU path).
    pub const DISPLAY: Budget = Budget { percent: 5, min: 64 * MIB, max: 768 * MIB };
    /// Frames uploaded to the GPU for drawing (on integrated GPUs, system RAM too).
    pub const GPU_UPLOADS: Budget = Budget { percent: 3, min: 96 * MIB, max: 512 * MIB };
    /// Idle recycled buffers ([`crate::pool`]), per kind.
    pub const IDLE_BUFFERS: Budget = Budget { percent: 2, min: 64 * MIB, max: 320 * MIB };

    /// The bytes this budget allows now.
    pub fn bytes(self) -> usize {
        self.bytes_for(app_budget(), under_pressure())
    }

    fn bytes_for(self, app: u64, pressure: bool) -> usize {
        let share = app / 100 * self.percent;
        let share = if pressure { share / 2 } else { share };
        usize::try_from(share.clamp(self.min, self.max.max(self.min))).unwrap_or(usize::MAX)
    }
}

/// What [`memory_status`] reports (`perf.stats`, the `memory.status` command).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryStatus {
    pub total: Option<u64>,
    pub available: Option<u64>,
    pub reserved: u64,
    pub app_budget: u64,
    pub under_pressure: bool,
}

pub fn memory_status() -> MemoryStatus {
    let m = system();
    MemoryStatus {
        total: m.map(|m| m.total),
        available: m.map(|m| m.available),
        reserved: reserved_bytes(),
        app_budget: app_budget(),
        under_pressure: under_pressure(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_is_parsed() {
        let text =
            "MemTotal:       15663104 kB\nMemFree:          562344 kB\nMemAvailable:    3738624 kB\nBuffers:          120000 kB\nCached:          3000000 kB\n";
        assert_eq!(parse_meminfo(text), Some(SystemMemory { total: 15_663_104 * 1024, available: 3_738_624 * 1024 }));
        // no MemAvailable (old kernels): free + cached
        let old = "MemTotal:  1000 kB\nMemFree:  100 kB\nCached:  200 kB\n";
        assert_eq!(parse_meminfo(old), Some(SystemMemory { total: 1_024_000, available: 307_200 }));
        // hostile text never panics
        for bad in [
            "",
            "MemTotal:",
            "MemTotal: x kB",
            "MemTotal: 0 kB\nMemAvailable: 5 kB",
            "MemTotal: 99999999999999999999 kB",
            "MemTotal: 10 kB\nMemAvailable: 999 kB",
        ] {
            let m = parse_meminfo(bad);
            assert!(m.is_none_or(|m| m.available <= m.total), "{bad:?}");
        }
    }

    #[test]
    fn the_app_budget_leaves_the_reserve() {
        assert_eq!(app_budget_for(16 * GIB, 4 * GIB), 12 * GIB);
        // a reserve larger than the machine still leaves enough to play
        assert_eq!(app_budget_for(16 * GIB, 64 * GIB), MIN_APP_BUDGET);
        // ... but never more than three quarters of a tiny machine
        assert_eq!(app_budget_for(GIB, 64 * GIB), 3 * GIB / 4);
        assert_eq!(app_budget_for(0, 0), 0);
    }

    #[test]
    fn budgets_scale_clamp_and_halve_under_pressure() {
        let b = Budget::DECODER;
        // a 16 GB machine with 4 GB reserved: 4 % of 12 GB
        let app = 12 * GIB;
        assert_eq!(b.bytes_for(app, false) as u64, app / 100 * 4);
        assert_eq!(b.bytes_for(app, true) as u64, (app / 100 * 4) / 2);
        // never below the minimum, never above the maximum
        assert_eq!(b.bytes_for(GIB, false) as u64, b.min);
        assert_eq!(b.bytes_for(GIB, true) as u64, b.min);
        assert_eq!(b.bytes_for(1024 * GIB, false) as u64, b.max);
        // crossing bounds do not panic
        assert_eq!(Budget { percent: 50, min: 10, max: 1 }.bytes_for(1000, false), 10);
    }

    #[test]
    fn pressure_is_a_tenth_of_ram_or_a_gigabyte() {
        assert!(pressure_for(SystemMemory { total: 16 * GIB, available: GIB }));
        assert!(!pressure_for(SystemMemory { total: 16 * GIB, available: 2 * GIB }));
        assert!(pressure_for(SystemMemory { total: 4 * GIB, available: GIB - 1 }));
        assert!(!pressure_for(SystemMemory { total: 4 * GIB, available: GIB }));
    }
}
