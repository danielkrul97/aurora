//! WGSL compute kernels for the CfC Liquid Neural Network cell.
//!
//! Forward kernels (one workgroup per batch element, threads stride
//! across neurons; T-loop runs inside the kernel using workgroup-shared
//! state):
//! - `lnn_forward_layer_zero`: layer 0, sparse one-hot token input.
//! - `lnn_forward_layer_deep`: layer 1+, dense input from previous
//!   layer's `x_packed`.
//!
//! Output stage:
//! - `lnn_output_proj`: dense GEMV `logits = b_o + W_out · x_top` per
//!   timestep.
//! - `softmax_cross_entropy`: per-timestep workgroup-shared softmax
//!   producing `d_logits = softmax - one_hot(target)` plus per-token
//!   loss.
//!
//! Backward kernels (mirrors of forward, atomic-free using per-batch
//! gradient buffers + a separate reduction kernel):
//! - `lnn_topproj_backward`: `d_x_top[b,t,:] = W_out^T · d_logits` plus
//!   per-batch `w_out` and `b_o` grads.
//! - `lnn_backward_layer_zero` / `lnn_backward_layer_deep`: per-layer
//!   reverse-time BPTT — 7 grad terms per neuron per timestep, weight
//!   grads accumulate into per-batch slices, recurrent `d_x_carry`
//!   propagated in workgroup memory.
//! - `reduce_per_batch`: sums the leading B axis of a per-batch grad
//!   buffer into a flat single-copy buffer.
//!
//! Param utility:
//! - `adam_step`: vectorized Adam over a parameter buffer.
//! - `clip_scale`: `b[i] *= factor` for grad clipping.
//! - `zero_buffer`: `b[i] = 0` for `zero_grad`.
//! - `clamp_log_tau`: clamp into `[-LOG_TAU_CLAMP, +LOG_TAU_CLAMP]`
//!   after each Adam step.
//!
//! Per-workgroup memory budget (workgroup_size = 256, MAX_HIDDEN = 1024):
//! - 3 × MAX_HIDDEN × 4 B = 12 KB shared.

#![allow(clippy::module_name_repetitions)]

pub const MAX_HIDDEN: u32 = 1024;
pub const WORKGROUP_SIZE: u32 = 256;
pub const PROJ_WORKGROUP_SIZE: u32 = 64;

pub const FORWARD_LAYER_ZERO_WGSL: &str = r"
struct ForwardParams {
    n_in: u32,
    n_hidden: u32,
    n_timesteps: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> w_f: array<f32>;
@group(0) @binding(1) var<storage, read> w_g: array<f32>;
@group(0) @binding(2) var<storage, read> w_h: array<f32>;
@group(0) @binding(3) var<storage, read> b_f: array<f32>;
@group(0) @binding(4) var<storage, read> b_g: array<f32>;
@group(0) @binding(5) var<storage, read> b_h: array<f32>;
@group(0) @binding(6) var<storage, read> log_tau: array<f32>;

@group(1) @binding(0) var<storage, read> input_tokens: array<u32>;
@group(1) @binding(1) var<storage, read> initial_x: array<f32>;
@group(1) @binding(2) var<storage, read_write> f_packed: array<f32>;
@group(1) @binding(3) var<storage, read_write> g_packed: array<f32>;
@group(1) @binding(4) var<storage, read_write> h_packed: array<f32>;
@group(1) @binding(5) var<storage, read_write> alpha_packed: array<f32>;
@group(1) @binding(6) var<storage, read_write> x_packed: array<f32>;
@group(1) @binding(7) var<storage, read_write> final_x: array<f32>;

var<push_constant> p: ForwardParams;

const MAX_HIDDEN: u32 = 1024u;
var<workgroup> sh_x_prev: array<f32, MAX_HIDDEN>;
var<workgroup> sh_x_new: array<f32, MAX_HIDDEN>;

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wid.x;
    let tid = lid.x;
    let n_h = p.n_hidden;
    let n_in = p.n_in;
    let n_z = n_in + n_h;
    let T = p.n_timesteps;

    var i_init = tid;
    loop {
        if (i_init >= n_h) { break; }
        sh_x_prev[i_init] = initial_x[b * n_h + i_init];
        i_init = i_init + 256u;
    }
    workgroupBarrier();

    for (var t: u32 = 0u; t < T; t = t + 1u) {
        let token = input_tokens[b * T + t];

        var i = tid;
        loop {
            if (i >= n_h) { break; }
            let row = i * n_z;
            var f_pre = b_f[i] + w_f[row + token];
            var g_pre = b_g[i] + w_g[row + token];
            var h_pre = b_h[i] + w_h[row + token];

            for (var j: u32 = 0u; j < n_h; j = j + 1u) {
                let v = sh_x_prev[j];
                f_pre = f_pre + w_f[row + n_in + j] * v;
                g_pre = g_pre + w_g[row + n_in + j] * v;
                h_pre = h_pre + w_h[row + n_in + j] * v;
            }

            let f_v = sigmoid(f_pre);
            let g_v = tanh(g_pre);
            let h_v = tanh(h_pre);
            let tau = exp(log_tau[i]);
            let alpha_v = sigmoid(-f_v * tau);
            let x_v = alpha_v * g_v + (1.0 - alpha_v) * h_v;

            let idx = b * T * n_h + t * n_h + i;
            f_packed[idx] = f_v;
            g_packed[idx] = g_v;
            h_packed[idx] = h_v;
            alpha_packed[idx] = alpha_v;
            x_packed[idx] = x_v;
            sh_x_new[i] = x_v;

            i = i + 256u;
        }
        workgroupBarrier();

        var i2 = tid;
        loop {
            if (i2 >= n_h) { break; }
            sh_x_prev[i2] = sh_x_new[i2];
            i2 = i2 + 256u;
        }
        workgroupBarrier();
    }

    var i3 = tid;
    loop {
        if (i3 >= n_h) { break; }
        final_x[b * n_h + i3] = sh_x_prev[i3];
        i3 = i3 + 256u;
    }
}
";

pub const FORWARD_LAYER_DEEP_WGSL: &str = r"
struct ForwardParams {
    n_in: u32,
    n_hidden: u32,
    n_timesteps: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> w_f: array<f32>;
@group(0) @binding(1) var<storage, read> w_g: array<f32>;
@group(0) @binding(2) var<storage, read> w_h: array<f32>;
@group(0) @binding(3) var<storage, read> b_f: array<f32>;
@group(0) @binding(4) var<storage, read> b_g: array<f32>;
@group(0) @binding(5) var<storage, read> b_h: array<f32>;
@group(0) @binding(6) var<storage, read> log_tau: array<f32>;

@group(1) @binding(0) var<storage, read> prev_x: array<f32>;
@group(1) @binding(1) var<storage, read> initial_x: array<f32>;
@group(1) @binding(2) var<storage, read_write> f_packed: array<f32>;
@group(1) @binding(3) var<storage, read_write> g_packed: array<f32>;
@group(1) @binding(4) var<storage, read_write> h_packed: array<f32>;
@group(1) @binding(5) var<storage, read_write> alpha_packed: array<f32>;
@group(1) @binding(6) var<storage, read_write> x_packed: array<f32>;
@group(1) @binding(7) var<storage, read_write> final_x: array<f32>;

var<push_constant> p: ForwardParams;

const MAX_HIDDEN: u32 = 1024u;
var<workgroup> sh_x_prev: array<f32, MAX_HIDDEN>;
var<workgroup> sh_x_new: array<f32, MAX_HIDDEN>;
var<workgroup> sh_input: array<f32, MAX_HIDDEN>;

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wid.x;
    let tid = lid.x;
    let n_h = p.n_hidden;
    let n_in = p.n_in;
    let n_z = n_in + n_h;
    let T = p.n_timesteps;

    var i_init = tid;
    loop {
        if (i_init >= n_h) { break; }
        sh_x_prev[i_init] = initial_x[b * n_h + i_init];
        i_init = i_init + 256u;
    }
    workgroupBarrier();

    for (var t: u32 = 0u; t < T; t = t + 1u) {
        var ji = tid;
        loop {
            if (ji >= n_in) { break; }
            sh_input[ji] = prev_x[b * T * n_in + t * n_in + ji];
            ji = ji + 256u;
        }
        workgroupBarrier();

        var i = tid;
        loop {
            if (i >= n_h) { break; }
            let row = i * n_z;
            var f_pre = b_f[i];
            var g_pre = b_g[i];
            var h_pre = b_h[i];

            for (var j: u32 = 0u; j < n_in; j = j + 1u) {
                let v = sh_input[j];
                f_pre = f_pre + w_f[row + j] * v;
                g_pre = g_pre + w_g[row + j] * v;
                h_pre = h_pre + w_h[row + j] * v;
            }
            for (var j: u32 = 0u; j < n_h; j = j + 1u) {
                let v = sh_x_prev[j];
                f_pre = f_pre + w_f[row + n_in + j] * v;
                g_pre = g_pre + w_g[row + n_in + j] * v;
                h_pre = h_pre + w_h[row + n_in + j] * v;
            }

            let f_v = sigmoid(f_pre);
            let g_v = tanh(g_pre);
            let h_v = tanh(h_pre);
            let tau = exp(log_tau[i]);
            let alpha_v = sigmoid(-f_v * tau);
            let x_v = alpha_v * g_v + (1.0 - alpha_v) * h_v;

            let idx = b * T * n_h + t * n_h + i;
            f_packed[idx] = f_v;
            g_packed[idx] = g_v;
            h_packed[idx] = h_v;
            alpha_packed[idx] = alpha_v;
            x_packed[idx] = x_v;
            sh_x_new[i] = x_v;

            i = i + 256u;
        }
        workgroupBarrier();

        var i2 = tid;
        loop {
            if (i2 >= n_h) { break; }
            sh_x_prev[i2] = sh_x_new[i2];
            i2 = i2 + 256u;
        }
        workgroupBarrier();
    }

    var i3 = tid;
    loop {
        if (i3 >= n_h) { break; }
        final_x[b * n_h + i3] = sh_x_prev[i3];
        i3 = i3 + 256u;
    }
}
";

pub const OUTPUT_PROJ_WGSL: &str = r"
struct ProjParams {
    n_hidden: u32,
    n_output: u32,
    n_timesteps: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> w_out: array<f32>;
@group(0) @binding(1) var<storage, read> b_o: array<f32>;
@group(0) @binding(2) var<storage, read> x_top: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;

var<push_constant> p: ProjParams;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bt = wid.x;
    let tid = lid.x;
    let n_h = p.n_hidden;
    let n_out = p.n_output;

    var v = tid;
    loop {
        if (v >= n_out) { break; }
        var acc = b_o[v];
        let row = v * n_h;
        let x_off = bt * n_h;
        for (var k: u32 = 0u; k < n_h; k = k + 1u) {
            acc = acc + w_out[row + k] * x_top[x_off + k];
        }
        logits[bt * n_out + v] = acc;
        v = v + 64u;
    }
}
";

pub const SOFTMAX_CE_WGSL: &str = r"
struct CeParams {
    n_output: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> d_logits: array<f32>;
@group(0) @binding(3) var<storage, read_write> losses: array<f32>;

var<push_constant> p: CeParams;

const WG: u32 = 256u;
var<workgroup> sh_max: array<f32, WG>;
var<workgroup> sh_sum: array<f32, WG>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bt = wid.x;
    let tid = lid.x;
    let V = p.n_output;
    let tgt_idx = targets[bt];
    let base = bt * V;

    var local_max = -1.0e30;
    var v = tid;
    loop {
        if (v >= V) { break; }
        local_max = max(local_max, logits[base + v]);
        v = v + 256u;
    }
    sh_max[tid] = local_max;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_max[tid] = max(sh_max[tid], sh_max[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let m = sh_max[0];

    var local_sum = 0.0;
    v = tid;
    loop {
        if (v >= V) { break; }
        local_sum = local_sum + exp(logits[base + v] - m);
        v = v + 256u;
    }
    sh_sum[tid] = local_sum;
    workgroupBarrier();

    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_sum[tid] = sh_sum[tid] + sh_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let total = sh_sum[0];

    v = tid;
    loop {
        if (v >= V) { break; }
        let l = logits[base + v];
        let p_v = exp(l - m) / total;
        var d = p_v;
        if (v == tgt_idx) { d = d - 1.0; }
        d_logits[base + v] = d;
        v = v + 256u;
    }

    if (tid == 0u) {
        let log_total = log(total);
        let loss_v = m + log_total - logits[base + tgt_idx];
        losses[bt] = loss_v;
    }
}
";

pub const TOPPROJ_BACKWARD_WGSL: &str = r"
struct TopProjBackParams {
    n_hidden: u32,
    n_output: u32,
    n_timesteps: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> w_out: array<f32>;
@group(0) @binding(1) var<storage, read> d_logits: array<f32>;
@group(0) @binding(2) var<storage, read> x_top: array<f32>;
@group(0) @binding(3) var<storage, read_write> d_x_top: array<f32>;
@group(0) @binding(4) var<storage, read_write> w_out_grad_pb: array<f32>;
@group(0) @binding(5) var<storage, read_write> b_o_grad_pb: array<f32>;

var<push_constant> p: TopProjBackParams;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wid.x;
    let tid = lid.x;
    let n_h = p.n_hidden;
    let n_out = p.n_output;
    let T = p.n_timesteps;

    // d_x_top[b, t, k] = sum_v w_out[v, k] * d_logits[b, t, v]
    for (var t: u32 = 0u; t < T; t = t + 1u) {
        var k = tid;
        loop {
            if (k >= n_h) { break; }
            var acc = 0.0;
            let dl_off = b * T * n_out + t * n_out;
            for (var v: u32 = 0u; v < n_out; v = v + 1u) {
                acc = acc + w_out[v * n_h + k] * d_logits[dl_off + v];
            }
            d_x_top[b * T * n_h + t * n_h + k] = acc;
            k = k + 256u;
        }
    }

    // w_out_grad[b, v, k] = sum_t d_logits[b, t, v] * x_top[b, t, k]
    // Store in per-batch buffer: [B, V, H] flat.
    var v_idx = tid;
    loop {
        if (v_idx >= n_out) { break; }
        let dl_v_base = b * T * n_out;
        var acc_b = 0.0;
        for (var t: u32 = 0u; t < T; t = t + 1u) {
            acc_b = acc_b + d_logits[dl_v_base + t * n_out + v_idx];
        }
        b_o_grad_pb[b * n_out + v_idx] = acc_b;
        v_idx = v_idx + 256u;
    }

    // For w_out_grad: distribute (v, k) pairs across threads.
    let total_cells = n_out * n_h;
    var cell = tid;
    loop {
        if (cell >= total_cells) { break; }
        let v = cell / n_h;
        let k = cell - v * n_h;
        var acc = 0.0;
        for (var t: u32 = 0u; t < T; t = t + 1u) {
            acc = acc + d_logits[b * T * n_out + t * n_out + v]
                * x_top[b * T * n_h + t * n_h + k];
        }
        w_out_grad_pb[b * n_out * n_h + v * n_h + k] = acc;
        cell = cell + 256u;
    }
}
";

pub const BACKWARD_LAYER_ZERO_WGSL: &str = r"
struct BackParams {
    n_in: u32,
    n_hidden: u32,
    n_timesteps: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> w_f: array<f32>;
@group(0) @binding(1) var<storage, read> w_g: array<f32>;
@group(0) @binding(2) var<storage, read> w_h: array<f32>;
@group(0) @binding(3) var<storage, read> log_tau: array<f32>;

@group(1) @binding(0) var<storage, read> input_tokens: array<u32>;
@group(1) @binding(1) var<storage, read> initial_x: array<f32>;
@group(1) @binding(2) var<storage, read> f_packed: array<f32>;
@group(1) @binding(3) var<storage, read> g_packed: array<f32>;
@group(1) @binding(4) var<storage, read> h_packed: array<f32>;
@group(1) @binding(5) var<storage, read> alpha_packed: array<f32>;
@group(1) @binding(6) var<storage, read> x_packed: array<f32>;
@group(1) @binding(7) var<storage, read> d_x_above: array<f32>;

@group(2) @binding(0) var<storage, read_write> w_f_grad_pb: array<f32>;
@group(2) @binding(1) var<storage, read_write> w_g_grad_pb: array<f32>;
@group(2) @binding(2) var<storage, read_write> w_h_grad_pb: array<f32>;
@group(2) @binding(3) var<storage, read_write> b_f_grad_pb: array<f32>;
@group(2) @binding(4) var<storage, read_write> b_g_grad_pb: array<f32>;
@group(2) @binding(5) var<storage, read_write> b_h_grad_pb: array<f32>;
@group(2) @binding(6) var<storage, read_write> log_tau_grad_pb: array<f32>;

var<push_constant> p: BackParams;

const MAX_HIDDEN: u32 = 1024u;
var<workgroup> sh_d_x_carry: array<f32, MAX_HIDDEN>;
var<workgroup> sh_d_x: array<f32, MAX_HIDDEN>;
var<workgroup> sh_d_f_pre: array<f32, MAX_HIDDEN>;
var<workgroup> sh_d_g_pre: array<f32, MAX_HIDDEN>;
var<workgroup> sh_d_h_pre: array<f32, MAX_HIDDEN>;
var<workgroup> sh_x_prev: array<f32, MAX_HIDDEN>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wid.x;
    let tid = lid.x;
    let n_h = p.n_hidden;
    let n_in = p.n_in;
    let n_z = n_in + n_h;
    let T = p.n_timesteps;

    // Initialize carry to zero.
    var i_init = tid;
    loop {
        if (i_init >= n_h) { break; }
        sh_d_x_carry[i_init] = 0.0;
        i_init = i_init + 256u;
    }
    workgroupBarrier();

    // Reverse-time loop.
    var t_back: u32 = T;
    loop {
        if (t_back == 0u) { break; }
        let t = t_back - 1u;

        // d_x[i] = d_x_above[b, t, i] + d_x_carry[i]
        var i_dx = tid;
        loop {
            if (i_dx >= n_h) { break; }
            sh_d_x[i_dx] = d_x_above[b * T * n_h + t * n_h + i_dx]
                + sh_d_x_carry[i_dx];
            i_dx = i_dx + 256u;
        }

        // Pre-load x_prev[t-1] (or initial state) into shared.
        var i_xp = tid;
        loop {
            if (i_xp >= n_h) { break; }
            if (t == 0u) {
                sh_x_prev[i_xp] = initial_x[b * n_h + i_xp];
            } else {
                sh_x_prev[i_xp] = x_packed[b * T * n_h + (t - 1u) * n_h + i_xp];
            }
            i_xp = i_xp + 256u;
        }
        workgroupBarrier();

        // Compute d_f_pre, d_g_pre, d_h_pre, accumulate b_*_grad and log_tau_grad.
        var i_pre = tid;
        loop {
            if (i_pre >= n_h) { break; }
            let act_idx = b * T * n_h + t * n_h + i_pre;
            let f_v = f_packed[act_idx];
            let g_v = g_packed[act_idx];
            let h_v = h_packed[act_idx];
            let alpha_v = alpha_packed[act_idx];
            let tau_v = exp(log_tau[i_pre]);
            let dx_v = sh_d_x[i_pre];

            let d_alpha = dx_v * (g_v - h_v);
            let d_g = dx_v * alpha_v;
            let d_h = dx_v * (1.0 - alpha_v);

            let dgp = d_g * (1.0 - g_v * g_v);
            let dhp = d_h * (1.0 - h_v * h_v);

            let d_arg = d_alpha * alpha_v * (1.0 - alpha_v);
            let d_f = d_arg * (-tau_v);
            let d_tau = d_arg * (-f_v);
            let dlt = d_tau * tau_v;
            log_tau_grad_pb[b * n_h + i_pre] = log_tau_grad_pb[b * n_h + i_pre] + dlt;

            let dfp = d_f * f_v * (1.0 - f_v);

            sh_d_f_pre[i_pre] = dfp;
            sh_d_g_pre[i_pre] = dgp;
            sh_d_h_pre[i_pre] = dhp;

            b_f_grad_pb[b * n_h + i_pre] = b_f_grad_pb[b * n_h + i_pre] + dfp;
            b_g_grad_pb[b * n_h + i_pre] = b_g_grad_pb[b * n_h + i_pre] + dgp;
            b_h_grad_pb[b * n_h + i_pre] = b_h_grad_pb[b * n_h + i_pre] + dhp;

            i_pre = i_pre + 256u;
        }
        workgroupBarrier();

        // Layer 0 input portion: only the token's column gets a grad.
        // Each thread handles a stripe of neurons updating that column.
        let token = input_tokens[b * T + t];
        var i_tok = tid;
        loop {
            if (i_tok >= n_h) { break; }
            let row = i_tok * n_z;
            let pb_off = b * n_h * n_z + row;
            w_f_grad_pb[pb_off + token] = w_f_grad_pb[pb_off + token] + sh_d_f_pre[i_tok];
            w_g_grad_pb[pb_off + token] = w_g_grad_pb[pb_off + token] + sh_d_g_pre[i_tok];
            w_h_grad_pb[pb_off + token] = w_h_grad_pb[pb_off + token] + sh_d_h_pre[i_tok];
            i_tok = i_tok + 256u;
        }

        // Recurrent portion: each thread handles one j (column of W_*[:, n_in+j]).
        // Computes weight grad (sum over i) and partial d_x_carry contribution.
        var j = tid;
        loop {
            if (j >= n_h) { break; }
            let v = sh_x_prev[j];
            var acc_carry = 0.0;
            for (var i: u32 = 0u; i < n_h; i = i + 1u) {
                let row = i * n_z;
                let pb_off = b * n_h * n_z + row + n_in + j;
                let dfp = sh_d_f_pre[i];
                let dgp = sh_d_g_pre[i];
                let dhp = sh_d_h_pre[i];
                w_f_grad_pb[pb_off] = w_f_grad_pb[pb_off] + dfp * v;
                w_g_grad_pb[pb_off] = w_g_grad_pb[pb_off] + dgp * v;
                w_h_grad_pb[pb_off] = w_h_grad_pb[pb_off] + dhp * v;
                acc_carry = acc_carry
                    + w_f[row + n_in + j] * dfp
                    + w_g[row + n_in + j] * dgp
                    + w_h[row + n_in + j] * dhp;
            }
            sh_d_x_carry[j] = acc_carry;
            j = j + 256u;
        }
        workgroupBarrier();

        t_back = t_back - 1u;
    }
}
";

pub const BACKWARD_LAYER_DEEP_WGSL: &str = r"
struct BackParams {
    n_in: u32,
    n_hidden: u32,
    n_timesteps: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> w_f: array<f32>;
@group(0) @binding(1) var<storage, read> w_g: array<f32>;
@group(0) @binding(2) var<storage, read> w_h: array<f32>;
@group(0) @binding(3) var<storage, read> log_tau: array<f32>;

@group(1) @binding(0) var<storage, read> prev_x: array<f32>;
@group(1) @binding(1) var<storage, read> initial_x: array<f32>;
@group(1) @binding(2) var<storage, read> f_packed: array<f32>;
@group(1) @binding(3) var<storage, read> g_packed: array<f32>;
@group(1) @binding(4) var<storage, read> h_packed: array<f32>;
@group(1) @binding(5) var<storage, read> alpha_packed: array<f32>;
@group(1) @binding(6) var<storage, read> x_packed: array<f32>;
@group(1) @binding(7) var<storage, read> d_x_above: array<f32>;

@group(2) @binding(0) var<storage, read_write> w_f_grad_pb: array<f32>;
@group(2) @binding(1) var<storage, read_write> w_g_grad_pb: array<f32>;
@group(2) @binding(2) var<storage, read_write> w_h_grad_pb: array<f32>;
@group(2) @binding(3) var<storage, read_write> b_f_grad_pb: array<f32>;
@group(2) @binding(4) var<storage, read_write> b_g_grad_pb: array<f32>;
@group(2) @binding(5) var<storage, read_write> b_h_grad_pb: array<f32>;
@group(2) @binding(6) var<storage, read_write> log_tau_grad_pb: array<f32>;
@group(2) @binding(7) var<storage, read_write> d_input_per_t: array<f32>;

var<push_constant> p: BackParams;

const MAX_HIDDEN: u32 = 1024u;
var<workgroup> sh_d_x_carry: array<f32, MAX_HIDDEN>;
var<workgroup> sh_d_x: array<f32, MAX_HIDDEN>;
var<workgroup> sh_d_f_pre: array<f32, MAX_HIDDEN>;
var<workgroup> sh_d_g_pre: array<f32, MAX_HIDDEN>;
var<workgroup> sh_d_h_pre: array<f32, MAX_HIDDEN>;
var<workgroup> sh_x_prev: array<f32, MAX_HIDDEN>;
var<workgroup> sh_input: array<f32, MAX_HIDDEN>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wid.x;
    let tid = lid.x;
    let n_h = p.n_hidden;
    let n_in = p.n_in;
    let n_z = n_in + n_h;
    let T = p.n_timesteps;

    var i_init = tid;
    loop {
        if (i_init >= n_h) { break; }
        sh_d_x_carry[i_init] = 0.0;
        i_init = i_init + 256u;
    }
    workgroupBarrier();

    var t_back: u32 = T;
    loop {
        if (t_back == 0u) { break; }
        let t = t_back - 1u;

        // d_x[i] = d_x_above[b, t, i] + d_x_carry[i]
        var i_dx = tid;
        loop {
            if (i_dx >= n_h) { break; }
            sh_d_x[i_dx] = d_x_above[b * T * n_h + t * n_h + i_dx]
                + sh_d_x_carry[i_dx];
            i_dx = i_dx + 256u;
        }

        var i_xp = tid;
        loop {
            if (i_xp >= n_h) { break; }
            if (t == 0u) {
                sh_x_prev[i_xp] = initial_x[b * n_h + i_xp];
            } else {
                sh_x_prev[i_xp] = x_packed[b * T * n_h + (t - 1u) * n_h + i_xp];
            }
            i_xp = i_xp + 256u;
        }

        var ji = tid;
        loop {
            if (ji >= n_in) { break; }
            sh_input[ji] = prev_x[b * T * n_in + t * n_in + ji];
            ji = ji + 256u;
        }
        workgroupBarrier();

        var i_pre = tid;
        loop {
            if (i_pre >= n_h) { break; }
            let act_idx = b * T * n_h + t * n_h + i_pre;
            let f_v = f_packed[act_idx];
            let g_v = g_packed[act_idx];
            let h_v = h_packed[act_idx];
            let alpha_v = alpha_packed[act_idx];
            let tau_v = exp(log_tau[i_pre]);
            let dx_v = sh_d_x[i_pre];

            let d_alpha = dx_v * (g_v - h_v);
            let d_g = dx_v * alpha_v;
            let d_h = dx_v * (1.0 - alpha_v);
            let dgp = d_g * (1.0 - g_v * g_v);
            let dhp = d_h * (1.0 - h_v * h_v);
            let d_arg = d_alpha * alpha_v * (1.0 - alpha_v);
            let d_f = d_arg * (-tau_v);
            let d_tau = d_arg * (-f_v);
            log_tau_grad_pb[b * n_h + i_pre] = log_tau_grad_pb[b * n_h + i_pre]
                + d_tau * tau_v;
            let dfp = d_f * f_v * (1.0 - f_v);

            sh_d_f_pre[i_pre] = dfp;
            sh_d_g_pre[i_pre] = dgp;
            sh_d_h_pre[i_pre] = dhp;

            b_f_grad_pb[b * n_h + i_pre] = b_f_grad_pb[b * n_h + i_pre] + dfp;
            b_g_grad_pb[b * n_h + i_pre] = b_g_grad_pb[b * n_h + i_pre] + dgp;
            b_h_grad_pb[b * n_h + i_pre] = b_h_grad_pb[b * n_h + i_pre] + dhp;

            i_pre = i_pre + 256u;
        }
        workgroupBarrier();

        // Input portion (dense): each thread = one j ∈ [0, n_in).
        var j_in = tid;
        loop {
            if (j_in >= n_in) { break; }
            let v = sh_input[j_in];
            var acc_di = 0.0;
            for (var i: u32 = 0u; i < n_h; i = i + 1u) {
                let row = i * n_z;
                let pb_off = b * n_h * n_z + row + j_in;
                let dfp = sh_d_f_pre[i];
                let dgp = sh_d_g_pre[i];
                let dhp = sh_d_h_pre[i];
                w_f_grad_pb[pb_off] = w_f_grad_pb[pb_off] + dfp * v;
                w_g_grad_pb[pb_off] = w_g_grad_pb[pb_off] + dgp * v;
                w_h_grad_pb[pb_off] = w_h_grad_pb[pb_off] + dhp * v;
                acc_di = acc_di
                    + w_f[row + j_in] * dfp
                    + w_g[row + j_in] * dgp
                    + w_h[row + j_in] * dhp;
            }
            d_input_per_t[b * T * n_in + t * n_in + j_in] = acc_di;
            j_in = j_in + 256u;
        }

        // Recurrent portion: each thread = one j ∈ [0, n_h).
        var j_r = tid;
        loop {
            if (j_r >= n_h) { break; }
            let v = sh_x_prev[j_r];
            var acc_carry = 0.0;
            for (var i: u32 = 0u; i < n_h; i = i + 1u) {
                let row = i * n_z;
                let pb_off = b * n_h * n_z + row + n_in + j_r;
                let dfp = sh_d_f_pre[i];
                let dgp = sh_d_g_pre[i];
                let dhp = sh_d_h_pre[i];
                w_f_grad_pb[pb_off] = w_f_grad_pb[pb_off] + dfp * v;
                w_g_grad_pb[pb_off] = w_g_grad_pb[pb_off] + dgp * v;
                w_h_grad_pb[pb_off] = w_h_grad_pb[pb_off] + dhp * v;
                acc_carry = acc_carry
                    + w_f[row + n_in + j_r] * dfp
                    + w_g[row + n_in + j_r] * dgp
                    + w_h[row + n_in + j_r] * dhp;
            }
            sh_d_x_carry[j_r] = acc_carry;
            j_r = j_r + 256u;
        }
        workgroupBarrier();

        t_back = t_back - 1u;
    }
}
";

pub const REDUCE_PER_BATCH_WGSL: &str = r"
struct ReduceParams {
    batch_size: u32,
    elem_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> per_batch: array<f32>;
@group(0) @binding(1) var<storage, read_write> total: array<f32>;

var<push_constant> p: ReduceParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.elem_count) { return; }
    var acc = 0.0;
    for (var b: u32 = 0u; b < p.batch_size; b = b + 1u) {
        acc = acc + per_batch[b * p.elem_count + idx];
    }
    total[idx] = total[idx] + acc;
}
";

pub const ZERO_BUFFER_WGSL: &str = r"
struct ZeroParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
var<push_constant> p: ZeroParams;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    buf[i] = 0.0;
}
";

pub const SCALE_BUFFER_WGSL: &str = r"
struct ScaleParams {
    n: u32,
    factor: f32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
var<push_constant> p: ScaleParams;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    buf[i] = buf[i] * p.factor;
}
";

pub const ADAM_STEP_WGSL: &str = r"
struct AdamParams {
    n: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    _pad: u32,
};
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m1: array<f32>;
@group(0) @binding(3) var<storage, read_write> m2: array<f32>;
var<push_constant> p: AdamParams;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let g = grads[i];
    let mm1 = p.beta1 * m1[i] + (1.0 - p.beta1) * g;
    let mm2 = p.beta2 * m2[i] + (1.0 - p.beta2) * g * g;
    m1[i] = mm1;
    m2[i] = mm2;
    let m1_hat = mm1 / p.bc1;
    let m2_hat = mm2 / p.bc2;
    params[i] = params[i] - p.lr * m1_hat / (sqrt(m2_hat) + p.eps);
}
";

pub const CLAMP_LOG_TAU_WGSL: &str = r"
struct ClampParams {
    n: u32,
    bound: f32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
var<push_constant> p: ClampParams;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    buf[i] = clamp(buf[i], -p.bound, p.bound);
}
";

pub const L2_NORM_SQUARED_WGSL: &str = r"
struct L2Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<storage, read> buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial: array<f32>;
var<push_constant> p: L2Params;
const WG: u32 = 256u;
var<workgroup> sh: array<f32, WG>;
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tid = lid.x;
    let idx = gid.x;
    var v: f32 = 0.0;
    if (idx < p.n) {
        let x = buf[idx];
        v = x * x;
    }
    sh[tid] = v;
    workgroupBarrier();
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { sh[tid] = sh[tid] + sh[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        partial[wid.x] = sh[0];
    }
}
";
