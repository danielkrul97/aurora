//! Single GPU device + queue, plus cached compute pipelines.

use std::sync::Arc;

#[derive(Clone)]
pub struct GpuContext {
    pub instance: Arc<wgpu::Instance>,
    pub adapter: Arc<wgpu::Adapter>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    /// Initialize a wgpu instance + adapter + device. Picks the first
    /// high-performance adapter available; falls back to anything that
    /// can run compute shaders. Synchronous wrapper over the async wgpu
    /// API via `pollster::block_on` so callers don't need to be async.
    ///
    /// # Errors
    /// Returns an error if no compute-capable adapter is available or
    /// device creation fails.
    pub fn new() -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GpuInitError::NoAdapter)?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("aurora gpu device"),
                // PUSH_CONSTANTS lets kernels read params via
                // `var<push_constant>` instead of a per-dispatch uniform
                // buffer + bind group entry; saves ~100-200 µs per
                // dispatch on the Vulkan path.
                required_features: {
                    let mut f = wgpu::Features::PUSH_CONSTANTS;
                    // Opt-in timestamp profiling (diagnostic only — see
                    // gpu::profiler). INSIDE_ENCODERS is what lets us mark
                    // between dispatches (not just at pass boundaries).
                    let ts = wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
                    if adapter.features().contains(ts) {
                        f |= ts;
                    }
                    f
                },
                required_limits: {
                    let supported = adapter.limits();
                    let mut wanted = wgpu::Limits::default();
                    wanted = wanted.using_resolution(supported.clone());
                    // 32 bytes covers all current kernel param structs
                    // (largest is AdamStepParams: 8 × u32/f32 = 32 B).
                    wanted.max_push_constant_size = 32.max(supported.max_push_constant_size);
                    wanted
                },
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .map_err(GpuInitError::Device)?;
        Ok(Self {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    #[must_use]
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }
}

#[derive(Debug)]
pub enum GpuInitError {
    NoAdapter,
    Device(wgpu::RequestDeviceError),
}

impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no compute-capable wgpu adapter found"),
            Self::Device(e) => write!(f, "wgpu device creation failed: {e}"),
        }
    }
}

impl std::error::Error for GpuInitError {}
