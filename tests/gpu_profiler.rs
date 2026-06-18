//! Phase 0: validate the wgpu timestamp path on the running device.
//! Skips cleanly when no GPU / no TIMESTAMP_QUERY.

#![cfg(feature = "gpu")]

use aurora::gpu::{profiler, GpuContext};

#[test]
fn timestamp_profiler_times_a_copy() {
    let Ok(ctx) = GpuContext::new() else {
        eprintln!("skip — no GPU");
        return;
    };
    match profiler::probe_copy_ms(&ctx, 4 * 1024 * 1024) {
        Some(ms) => {
            eprintln!("gpu_profiler probe: 16 MB copy = {ms:.4} ms");
            assert!(ms > 0.0 && ms < 200.0, "copy span {ms} ms out of sane range");
        }
        None => eprintln!("skip — adapter lacks TIMESTAMP_QUERY"),
    }
}
