//! GPU feature probe: does this adapter expose SHADER_F16 (the gate for the
//! f16 conv path)? Run with `--features gpu`.

fn main() {
    let ctx = match aurora::gpu::GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };
    let f = ctx.adapter.features();
    println!("adapter        : {}", ctx.adapter_info().name);
    println!("SHADER_F16     : {}", f.contains(wgpu::Features::SHADER_F16));
    println!("PUSH_CONSTANTS : {}", f.contains(wgpu::Features::PUSH_CONSTANTS));
    println!("SUBGROUP       : {}", f.contains(wgpu::Features::SUBGROUP));

    if !f.contains(wgpu::Features::SHADER_F16) {
        println!("f16 pipeline   : skipped (no SHADER_F16)");
        return;
    }
    // Does naga actually compile f16 WGSL (enable f16 + vec2<f16> packed mul)?
    // Shader compilation happens at pipeline creation, so force it and catch
    // validation errors via the device error scope.
    let wgsl = "\
enable f16;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    let a = vec2<f16>(1.0h, 2.0h);
    let b = vec2<f16>(3.0h, 4.0h);
    let p = a * b;
    out[0] = f32(p.x) + f32(p.y);
}
";
    ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("f16 probe"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let _pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("f16 probe"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let err = pollster::block_on(ctx.device.pop_error_scope());
    match err {
        Some(e) => println!("f16 pipeline   : FAILED — {e}"),
        None => println!("f16 pipeline   : OK (naga compiles f16 packed math)"),
    }
}
