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
                required_features: wgpu::Features::empty(),
                required_limits: {
                    // Default desktop limits give us 8+ storage bindings per
                    // stage, which the gated_cell_step kernel needs (9). The
                    // fallback `downlevel_defaults` caps at 4.
                    let supported = adapter.limits();
                    let mut wanted = wgpu::Limits::default();
                    wanted = wanted.using_resolution(supported.clone());
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
