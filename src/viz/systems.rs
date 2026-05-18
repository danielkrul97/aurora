use bevy::prelude::*;
use bevy::sprite_render::{ColorMaterial, MeshMaterial2d};

use super::components::{
    AttnBar, EdgeFrom, FfnBar, LastStepTick, LayerBox, LossChart, Particle, PulseState,
    SnapshotHandle, VizAssets,
};

#[derive(Resource, Debug, Clone, Default)]
pub struct LiveSnapshot(pub Option<super::snapshot::TrainingSnapshot>);

pub fn poll_snapshot(handle: Res<SnapshotHandle>, mut live: ResMut<LiveSnapshot>) {
    if let Ok(snap) = handle.0.try_lock() {
        live.0 = Some(snap.clone());
    }
}

fn health_to_color(h: f32) -> Color {
    let h = h.clamp(0.0, 1.0);
    if h < 0.5 {
        let t = h * 2.0;
        Color::srgb(
            0.78 * (1.0 - t) + 0.90 * t,
            0.22 * (1.0 - t) + 0.62 * t,
            0.22 * (1.0 - t) + 0.20 * t,
        )
    } else {
        let t = (h - 0.5) * 2.0;
        Color::srgb(
            0.90 * (1.0 - t) + 0.30 * t,
            0.62 * (1.0 - t) + 0.62 * t,
            0.20 * (1.0 - t) + 0.36 * t,
        )
    }
}

pub fn update_layer_colors(
    live: Res<LiveSnapshot>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    q: Query<&LayerBox>,
) {
    let Some(snap) = live.0.as_ref() else {
        return;
    };
    for lb in &q {
        let Some(stage) = snap.layers.get(lb.stage) else {
            continue;
        };
        let c = health_to_color(stage.health);
        if let Some(mat) = materials.get_mut(&lb.material) {
            mat.color = c;
        }
    }
}

pub fn update_attn_bars(
    live: Res<LiveSnapshot>,
    mut q: Query<(&AttnBar, &mut Transform)>,
) {
    let Some(snap) = live.0.as_ref() else {
        return;
    };
    for (bar, mut tf) in &mut q {
        let Some(stage) = snap.layers.get(bar.stage) else {
            continue;
        };
        let v = stage.attn_bar.clamp(0.0, 1.0);
        let target_w = (v * bar.max_w).max(1.0);
        let cur_w = tf.scale.x.max(1.0);
        let new_w = lerp(cur_w, target_w, 0.18);
        tf.scale.x = new_w;
        tf.scale.y = bar.height;
        tf.translation.x = bar.anchor_x + new_w * 0.5;
    }
}

pub fn update_ffn_bars(
    live: Res<LiveSnapshot>,
    mut q: Query<(&FfnBar, &mut Transform)>,
) {
    let Some(snap) = live.0.as_ref() else {
        return;
    };
    for (bar, mut tf) in &mut q {
        let Some(stage) = snap.layers.get(bar.stage) else {
            continue;
        };
        let v = stage.ffn_bar.clamp(0.0, 1.0);
        let target_w = (v * bar.max_w).max(1.0);
        let cur_w = tf.scale.x.max(1.0);
        let new_w = lerp(cur_w, target_w, 0.18);
        tf.scale.x = new_w;
        tf.scale.y = bar.height;
        tf.translation.x = bar.anchor_x + new_w * 0.5;
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

const PULSE_DURATION_S: f32 = 0.28;
const PARTICLE_BASE_INTERVAL_S: f32 = 0.42;
const PARTICLE_MIN_INTERVAL_S: f32 = 0.14;
const PARTICLE_RADIUS: f32 = 4.5;

pub fn trigger_pulses(
    live: Res<LiveSnapshot>,
    mut last: ResMut<LastStepTick>,
    mut q: Query<&mut PulseState>,
) {
    let Some(snap) = live.0.as_ref() else {
        return;
    };
    if snap.step_tick != last.0 {
        last.0 = snap.step_tick;
        for mut p in &mut q {
            p.remaining_s = PULSE_DURATION_S;
            p.last_step_seen = snap.step_tick;
        }
    }
}

pub fn decay_pulses(time: Res<Time>, mut q: Query<&mut PulseState>) {
    let dt = time.delta_secs();
    for mut p in &mut q {
        if p.remaining_s > 0.0 {
            p.remaining_s = (p.remaining_s - dt).max(0.0);
        }
    }
}

pub fn pulse_layer_scale(mut q: Query<(&LayerBox, &PulseState, &mut Transform)>) {
    for (lb, pulse, mut tf) in &mut q {
        let boost = (pulse.remaining_s / PULSE_DURATION_S).clamp(0.0, 1.0);
        let scale_factor = 1.0 + boost * 0.045;
        tf.scale.x = lb.base_w * scale_factor;
        tf.scale.y = lb.base_h * scale_factor;
    }
}

pub fn spawn_particles(
    time: Res<Time>,
    live: Res<LiveSnapshot>,
    assets: Option<Res<VizAssets>>,
    mut commands: Commands,
    mut edges: Query<&mut EdgeFrom>,
) {
    let Some(assets) = assets else { return };
    let now = time.elapsed_secs();
    let loss = live.0.as_ref().map_or(1.0, |s| s.loss.max(0.05));
    let interval = (PARTICLE_BASE_INTERVAL_S / (1.0 + 1.0 / loss).max(0.5))
        .clamp(PARTICLE_MIN_INTERVAL_S, PARTICLE_BASE_INTERVAL_S);
    let speed = (1.4 / (loss + 0.5)).clamp(0.35, 1.6);
    let size = PARTICLE_RADIUS * 2.0;

    for mut edge in &mut edges {
        if now - edge.last_emit_s >= interval {
            edge.last_emit_s = now;
            let start = edge.start_pos;
            let end = edge.end_pos;
            commands.spawn((
                Mesh2d(assets.unit_rect.clone()),
                MeshMaterial2d(assets.particle_mat.clone()),
                Transform {
                    translation: Vec3::new(start.x, start.y, 0.4),
                    scale: Vec3::new(size, size, 1.0),
                    ..default()
                },
                Particle {
                    progress: 0.0,
                    speed,
                    start,
                    end,
                },
            ));
        }
    }
}

pub fn advance_particles(
    time: Res<Time>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut Particle, &mut Transform)>,
) {
    let dt = time.delta_secs();
    let size = PARTICLE_RADIUS * 2.0;
    for (entity, mut p, mut tf) in &mut q {
        p.progress += p.speed * dt;
        if p.progress >= 1.0 {
            commands.entity(entity).despawn();
            continue;
        }
        let pos = p.start + (p.end - p.start) * p.progress;
        tf.translation.x = pos.x;
        tf.translation.y = pos.y;
        let fade = if p.progress < 0.15 {
            p.progress / 0.15
        } else if p.progress > 0.85 {
            (1.0 - p.progress) / 0.15
        } else {
            1.0
        };
        let s = size * (0.85 + 0.15 * fade);
        tf.scale.x = s;
        tf.scale.y = s;
    }
}

pub fn draw_loss_chart(
    live: Res<LiveSnapshot>,
    chart_q: Query<&LossChart>,
    mut gizmos: Gizmos,
) {
    let Some(snap) = live.0.as_ref() else {
        return;
    };
    let Ok(chart) = chart_q.single() else {
        return;
    };
    let history = &snap.loss_history;
    if history.len() < 2 {
        return;
    }

    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &(_, v) in history {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return;
    }
    let span = (hi - lo).max(1e-3);
    let half = chart.size * 0.5;
    let margin = Vec2::new(14.0, 22.0);
    let inner_w = chart.size.x - 2.0 * margin.x;
    let inner_h = chart.size.y - 2.0 * margin.y;
    let n = history.len();

    let tl = Vec2::new(chart.center.x - half.x + 4.0, chart.center.y + half.y - 4.0);
    let tr = Vec2::new(chart.center.x + half.x - 4.0, chart.center.y + half.y - 4.0);
    let bl = Vec2::new(chart.center.x - half.x + 4.0, chart.center.y - half.y + 4.0);
    let br = Vec2::new(chart.center.x + half.x - 4.0, chart.center.y - half.y + 4.0);
    let border = Color::srgba(0.30, 0.36, 0.46, 0.85);
    gizmos.line_2d(tl, tr, border);
    gizmos.line_2d(tr, br, border);
    gizmos.line_2d(br, bl, border);
    gizmos.line_2d(bl, tl, border);

    let mut prev: Option<Vec2> = None;
    for (i, &(_, v)) in history.iter().enumerate() {
        if !v.is_finite() {
            prev = None;
            continue;
        }
        let tx = i as f32 / (n - 1).max(1) as f32;
        let ty = (v - lo) / span;
        let x = chart.center.x - half.x + margin.x + tx * inner_w;
        let y = chart.center.y - half.y + margin.y + ty * inner_h;
        let cur = Vec2::new(x, y);
        if let Some(p) = prev {
            gizmos.line_2d(p, cur, Color::srgb(0.14, 0.50, 0.28));
        }
        prev = Some(cur);
    }

    let tick_x = chart.center.x + half.x - 6.0;
    let tick_color = Color::srgba(0.30, 0.36, 0.46, 0.9);
    gizmos.line_2d(
        Vec2::new(tick_x - 6.0, chart.center.y - half.y + margin.y),
        Vec2::new(tick_x, chart.center.y - half.y + margin.y),
        tick_color,
    );
    gizmos.line_2d(
        Vec2::new(tick_x - 6.0, chart.center.y + half.y - margin.y),
        Vec2::new(tick_x, chart.center.y + half.y - margin.y),
        tick_color,
    );
}
