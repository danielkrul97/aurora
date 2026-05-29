//! Live Bevy visualization of the thermodynamic-LM training run.
//!
//! Centerpiece is a grid of heatmaps of the evolving gas field
//! (Temperature, Density, Energy, |Velocity|) — the §7 interpretability
//! promise: you watch the field "breathe" as the model trains. Plus a
//! train-bpb curve with the bigram baseline drawn in, a status line, and
//! the input/prediction strip.
//!
//! Same threading contract as [`super::run_window`]: call
//! [`run_thermo_window`] on the OS main thread; it blocks until the window
//! closes. The training thread writes a [`ThermoSnapshot`] behind a mutex;
//! the render thread `try_lock`s and clones it each frame.
//!
//! Rendering uses the same `Mesh2d` + per-entity `ColorMaterial` mutation
//! idiom as the transformer viz (one material per heat cell), which is the
//! pattern proven to render in this Bevy build (`Sprite::from_color` does
//! not).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use bevy::sprite::Anchor;
use bevy::sprite_render::{ColorMaterial, MeshMaterial2d};

const HISTORY_CAP: usize = 600;

// ───── shared snapshot ──────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ThermoChannel {
    pub name: String,
    /// `[h*w]`, row-major.
    pub data: Vec<f32>,
    pub vmin: f32,
    pub vmax: f32,
}

#[derive(Clone, Debug)]
pub struct ThermoSnapshot {
    pub step: u64,
    pub total_steps: u64,
    pub loss: f32,
    pub train_bpb: f32,
    pub val_bpb: f32,
    pub best_val: f32,
    pub bigram_bpb: f32,
    pub lr: f32,
    pub grad_norm: f32,
    pub elapsed_s: f64,
    pub params: usize,
    pub h: usize,
    pub w: usize,
    pub k_steps: usize,
    pub dt: f32,
    pub channels: Vec<ThermoChannel>,
    pub input_preview: String,
    pub pred_preview: String,
    pub pred_correct: Option<bool>,
    pub bpb_history: VecDeque<(u64, f32)>,
    pub step_tick: u64,
    pub finished: bool,
}

impl ThermoSnapshot {
    #[must_use]
    pub fn new(
        h: usize,
        w: usize,
        k_steps: usize,
        dt: f32,
        params: usize,
        total_steps: u64,
        bigram_bpb: f32,
        channel_names: &[&str],
    ) -> Self {
        let channels = channel_names
            .iter()
            .map(|n| ThermoChannel {
                name: (*n).to_string(),
                data: vec![0.0; h * w],
                vmin: 0.0,
                vmax: 1.0,
            })
            .collect();
        Self {
            step: 0,
            total_steps,
            loss: 0.0,
            train_bpb: 0.0,
            val_bpb: f32::INFINITY,
            best_val: f32::INFINITY,
            bigram_bpb,
            lr: 0.0,
            grad_norm: 0.0,
            elapsed_s: 0.0,
            params,
            h,
            w,
            k_steps,
            dt,
            channels,
            input_preview: String::new(),
            pred_preview: String::new(),
            pred_correct: None,
            bpb_history: VecDeque::with_capacity(HISTORY_CAP),
            step_tick: 0,
            finished: false,
        }
    }

    pub fn push_bpb(&mut self, step: u64, bpb: f32) {
        if self.bpb_history.len() == HISTORY_CAP {
            self.bpb_history.pop_front();
        }
        self.bpb_history.push_back((step, bpb));
    }
}

pub type ThermoSnapshotArc = Arc<Mutex<ThermoSnapshot>>;

#[must_use]
pub fn new_thermo_snapshot(
    h: usize,
    w: usize,
    k_steps: usize,
    dt: f32,
    params: usize,
    total_steps: u64,
    bigram_bpb: f32,
    channel_names: &[&str],
) -> ThermoSnapshotArc {
    Arc::new(Mutex::new(ThermoSnapshot::new(
        h, w, k_steps, dt, params, total_steps, bigram_bpb, channel_names,
    )))
}

// ───── resources & components ───────────────────────────────────────────

#[derive(Resource)]
struct ThermoHandle(ThermoSnapshotArc);

#[derive(Resource, Default)]
struct ThermoLive(Option<ThermoSnapshot>);

#[derive(Resource, Clone)]
struct ThermoLayout {
    h: usize,
    w: usize,
    names: Vec<String>,
}

/// Geometry of each heatmap panel, for gizmo overlays.
#[derive(Resource, Default)]
struct ThermoPanels(Vec<PanelGeom>);

#[derive(Clone, Copy)]
struct PanelGeom {
    center: Vec2,
    side: f32,
}

#[derive(Component)]
struct HeatCell {
    channel: usize,
    index: usize,
    material: Handle<ColorMaterial>,
}

#[derive(Component)]
struct ThermoStatus;

#[derive(Component)]
struct ThermoSubtitle;

#[derive(Component)]
struct ThermoTokens;

#[derive(Component, Clone, Copy)]
struct BpbChart {
    center: Vec2,
    size: Vec2,
}

/// Frame counter for the optional one-shot screenshot (env `THERMO_VIZ_SHOT`).
#[derive(Resource, Default)]
struct ShotState {
    frame: u32,
    fired: bool,
}

// ───── entry point ──────────────────────────────────────────────────────

/// Run the thermo viz window. Blocks on the OS main thread until closed.
pub fn run_thermo_window(snapshot: ThermoSnapshotArc) {
    let (h, w, names) = {
        let s = snapshot.lock().expect("snapshot poisoned");
        (s.h, s.w, s.channels.iter().map(|c| c.name.clone()).collect::<Vec<_>>())
    };

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Aurora — thermo LM (gas substrate)".into(),
                        resolution: (1400u32, 900u32).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(LogPlugin {
                    filter: "wgpu=error,naga=warn,bevy_render::view::window=error".into(),
                    level: bevy::log::Level::INFO,
                    ..default()
                }),
        )
        .insert_resource(ClearColor(Color::srgb(0.07, 0.08, 0.11)))
        .insert_resource(ThermoHandle(snapshot))
        .insert_resource(ThermoLive::default())
        .insert_resource(ThermoLayout { h, w, names })
        .insert_resource(ThermoPanels::default())
        .insert_resource(ShotState::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                poll_snapshot,
                update_heatmaps,
                draw_inject_markers,
                draw_bpb_chart,
                refresh_status,
                refresh_subtitle,
                refresh_tokens,
                auto_screenshot,
            )
                .chain(),
        )
        .run();
}

// ───── layout constants ─────────────────────────────────────────────────

const STATUS_H: f32 = 64.0;
const TOKENS_H: f32 = 60.0;
const PANEL_BG: Color = Color::srgb(0.12, 0.13, 0.17);
const PANEL_BORDER: Color = Color::srgb(0.32, 0.36, 0.46);
const TITLE_COLOR: Color = Color::srgb(0.82, 0.86, 0.94);

fn rect_tf(cx: f32, cy: f32, w: f32, h: f32, z: f32) -> Transform {
    Transform {
        translation: Vec3::new(cx, cy, z),
        scale: Vec3::new(w, h, 1.0),
        ..default()
    }
}

// ───── startup scene ────────────────────────────────────────────────────

#[allow(clippy::needless_range_loop, clippy::cast_precision_loss)]
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    layout: Res<ThermoLayout>,
    mut panels: ResMut<ThermoPanels>,
    windows: Query<&Window>,
) {
    commands.spawn(Camera2d);
    let unit = meshes.add(Rectangle::new(1.0, 1.0));

    let (win_w, win_h) = windows
        .single()
        .map(|w| (w.resolution.width(), w.resolution.height()))
        .unwrap_or((1400.0, 900.0));

    let content_top = win_h * 0.5 - STATUS_H - 28.0;
    let content_bottom = -win_h * 0.5 + TOKENS_H + 14.0;
    let content_h = content_top - content_bottom;

    // Heatmaps occupy the left ~66%; the bpb chart sits on the right.
    let heat_left = -win_w * 0.5 + 24.0;
    let heat_right = win_w * 0.16;
    let heat_w = heat_right - heat_left;
    let (cols, rows) = (2usize, 2usize);
    let col_w = heat_w / cols as f32;
    let row_h = content_h / rows as f32;
    let side = (col_w.min(row_h) * 0.74).min(300.0);

    let n = layout.names.len().min(cols * rows);
    for c in 0..n {
        let (ci, ri) = (c % cols, c / cols);
        let pcx = heat_left + col_w * (ci as f32 + 0.5);
        let pcy = content_top - row_h * (ri as f32 + 0.5);
        panels.0.push(PanelGeom { center: Vec2::new(pcx, pcy), side });
        spawn_heatmap(&mut commands, &mut materials, &unit, &layout, c, pcx, pcy, side);
    }

    spawn_bpb_chart(&mut commands, &unit, &mut materials, win_w, content_top, content_bottom);
    spawn_text_bands(&mut commands, &unit, &mut materials, win_w, win_h);
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::cast_precision_loss)]
fn spawn_heatmap(
    commands: &mut Commands,
    materials: &mut Assets<ColorMaterial>,
    unit: &Handle<Mesh>,
    layout: &ThermoLayout,
    channel: usize,
    pcx: f32,
    pcy: f32,
    side: f32,
) {
    let (h, w) = (layout.h, layout.w);
    // border + background.
    commands.spawn((
        Mesh2d(unit.clone()),
        MeshMaterial2d(materials.add(ColorMaterial::from_color(PANEL_BORDER))),
        rect_tf(pcx, pcy, side + 6.0, side + 6.0, -0.5),
    ));
    commands.spawn((
        Mesh2d(unit.clone()),
        MeshMaterial2d(materials.add(ColorMaterial::from_color(PANEL_BG))),
        rect_tf(pcx, pcy, side, side, -0.2),
    ));

    let cell_w = side / w as f32;
    let cell_h = side / h as f32;
    for i in 0..h {
        for j in 0..w {
            let x = pcx - side * 0.5 + (j as f32 + 0.5) * cell_w;
            let y = pcy + side * 0.5 - (i as f32 + 0.5) * cell_h;
            let mat = materials.add(ColorMaterial::from_color(Color::srgb(0.05, 0.05, 0.1)));
            commands.spawn((
                Mesh2d(unit.clone()),
                MeshMaterial2d(mat.clone()),
                rect_tf(x, y, cell_w, cell_h, 0.0),
                HeatCell { channel, index: i * w + j, material: mat },
            ));
        }
    }

    commands.spawn((
        Text2d::new(layout.names[channel].clone()),
        TextFont { font_size: 16.0, ..default() },
        TextColor(TITLE_COLOR),
        Anchor::CENTER_LEFT,
        Transform::from_xyz(pcx - side * 0.5, pcy + side * 0.5 + 14.0, 1.0),
    ));
}

fn spawn_bpb_chart(
    commands: &mut Commands,
    unit: &Handle<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    win_w: f32,
    content_top: f32,
    content_bottom: f32,
) {
    let panel_w = win_w * 0.30;
    let panel_h = (content_top - content_bottom) * 0.6;
    let cx = win_w * 0.5 - panel_w * 0.5 - 28.0;
    let cy = content_top - panel_h * 0.5;
    commands.spawn((
        Mesh2d(unit.clone()),
        MeshMaterial2d(materials.add(ColorMaterial::from_color(PANEL_BG))),
        rect_tf(cx, cy, panel_w, panel_h, -0.5),
        BpbChart { center: Vec2::new(cx, cy), size: Vec2::new(panel_w, panel_h) },
    ));
    commands.spawn((
        Text2d::new("train bits/byte  (red = bigram baseline)"),
        TextFont { font_size: 14.0, ..default() },
        TextColor(TITLE_COLOR),
        Anchor::CENTER_LEFT,
        Transform::from_xyz(cx - panel_w * 0.5, cy + panel_h * 0.5 + 14.0, 1.0),
    ));
}

fn spawn_text_bands(
    commands: &mut Commands,
    unit: &Handle<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    win_w: f32,
    win_h: f32,
) {
    let band = Color::srgb(0.10, 0.11, 0.15);
    // status band (top)
    let sy = win_h * 0.5 - STATUS_H * 0.5;
    commands.spawn((
        Mesh2d(unit.clone()),
        MeshMaterial2d(materials.add(ColorMaterial::from_color(band))),
        rect_tf(0.0, sy, win_w, STATUS_H, 5.0),
    ));
    commands.spawn((
        Text2d::new("THERMO  |  waiting for first step…"),
        TextFont { font_size: 19.0, ..default() },
        TextColor(Color::srgb(0.86, 0.90, 0.96)),
        Anchor::CENTER_LEFT,
        Transform::from_xyz(-win_w * 0.5 + 22.0, sy + 12.0, 5.2),
        ThermoStatus,
    ));
    commands.spawn((
        Text2d::new(""),
        TextFont { font_size: 14.0, ..default() },
        TextColor(Color::srgb(0.62, 0.70, 0.82)),
        Anchor::CENTER_LEFT,
        Transform::from_xyz(-win_w * 0.5 + 22.0, sy - 12.0, 5.2),
        ThermoSubtitle,
    ));
    // tokens band (bottom)
    let ty = -win_h * 0.5 + TOKENS_H * 0.5;
    commands.spawn((
        Mesh2d(unit.clone()),
        MeshMaterial2d(materials.add(ColorMaterial::from_color(band))),
        rect_tf(0.0, ty, win_w, TOKENS_H, 5.0),
    ));
    commands.spawn((
        Text2d::new("input:  (no tokens yet)    ->  pred: -"),
        TextFont { font_size: 17.0, ..default() },
        TextColor(Color::srgb(0.74, 0.80, 0.90)),
        Anchor::CENTER_LEFT,
        Transform::from_xyz(-win_w * 0.5 + 22.0, ty, 5.2),
        ThermoTokens,
    ));
}

// ───── update systems ───────────────────────────────────────────────────

fn poll_snapshot(handle: Res<ThermoHandle>, mut live: ResMut<ThermoLive>) {
    if let Ok(s) = handle.0.try_lock() {
        live.0 = Some(s.clone());
    }
}

/// 5-stop thermal colormap (dark → purple → red → orange → bright).
fn thermal(t: f32) -> Color {
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.00, [0.03, 0.03, 0.12]),
        (0.25, [0.32, 0.05, 0.42]),
        (0.50, [0.76, 0.18, 0.18]),
        (0.75, [0.97, 0.58, 0.10]),
        (1.00, [0.99, 0.97, 0.82]),
    ];
    let t = t.clamp(0.0, 1.0);
    let mut k = 0;
    while k + 1 < STOPS.len() && t > STOPS[k + 1].0 {
        k += 1;
    }
    let (t0, c0) = STOPS[k];
    let (t1, c1) = STOPS[(k + 1).min(STOPS.len() - 1)];
    let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
    Color::srgb(
        c0[0] + (c1[0] - c0[0]) * f,
        c0[1] + (c1[1] - c0[1]) * f,
        c0[2] + (c1[2] - c0[2]) * f,
    )
}

fn update_heatmaps(
    live: Res<ThermoLive>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    q: Query<&HeatCell>,
) {
    let Some(snap) = live.0.as_ref() else { return };
    for cell in &q {
        let Some(ch) = snap.channels.get(cell.channel) else { continue };
        let Some(&v) = ch.data.get(cell.index) else { continue };
        let t = if ch.vmax > ch.vmin {
            (v - ch.vmin) / (ch.vmax - ch.vmin)
        } else {
            0.5
        };
        if let Some(mat) = materials.get_mut(&cell.material) {
            mat.color = thermal(t);
        }
    }
}

fn draw_inject_markers(panels: Res<ThermoPanels>, mut gizmos: Gizmos) {
    let c = Color::srgba(0.95, 0.97, 0.85, 0.55);
    for p in &panels.0 {
        let r = (p.side * 0.5 / 6.0).max(4.0); // matches the Gaussian nozzle σ
        gizmos.line_2d(p.center - Vec2::new(r, 0.0), p.center + Vec2::new(r, 0.0), c);
        gizmos.line_2d(p.center - Vec2::new(0.0, r), p.center + Vec2::new(0.0, r), c);
    }
}

#[allow(clippy::cast_precision_loss)]
fn draw_bpb_chart(live: Res<ThermoLive>, chart_q: Query<&BpbChart>, mut gizmos: Gizmos) {
    let Some(snap) = live.0.as_ref() else { return };
    let Ok(chart) = chart_q.single() else { return };
    let half = chart.size * 0.5;
    let margin = Vec2::new(16.0, 18.0);
    let inner_w = chart.size.x - 2.0 * margin.x;
    let inner_h = chart.size.y - 2.0 * margin.y;

    // y-range covers the history and the bigram baseline.
    let (mut lo, mut hi) = (snap.bigram_bpb, snap.bigram_bpb);
    for &(_, v) in &snap.bpb_history {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return;
    }
    let pad = (hi - lo).max(0.1) * 0.1;
    lo -= pad;
    hi += pad;
    let span = (hi - lo).max(1e-3);
    let map = |i: f32, n: f32, v: f32| -> Vec2 {
        let tx = if n > 1.0 { i / (n - 1.0) } else { 0.0 };
        let ty = (v - lo) / span;
        Vec2::new(
            chart.center.x - half.x + margin.x + tx * inner_w,
            chart.center.y - half.y + margin.y + ty * inner_h,
        )
    };

    // border
    let b = Color::srgba(0.34, 0.40, 0.50, 0.85);
    let tl = Vec2::new(chart.center.x - half.x + 4.0, chart.center.y + half.y - 4.0);
    let tr = Vec2::new(chart.center.x + half.x - 4.0, chart.center.y + half.y - 4.0);
    let bl = Vec2::new(chart.center.x - half.x + 4.0, chart.center.y - half.y + 4.0);
    let br = Vec2::new(chart.center.x + half.x - 4.0, chart.center.y - half.y + 4.0);
    gizmos.line_2d(tl, tr, b);
    gizmos.line_2d(tr, br, b);
    gizmos.line_2d(br, bl, b);
    gizmos.line_2d(bl, tl, b);

    // bigram baseline (red horizontal line)
    if snap.bigram_bpb.is_finite() {
        let yb = chart.center.y - half.y + margin.y + ((snap.bigram_bpb - lo) / span) * inner_h;
        gizmos.line_2d(
            Vec2::new(chart.center.x - half.x + margin.x, yb),
            Vec2::new(chart.center.x + half.x - margin.x, yb),
            Color::srgb(0.88, 0.24, 0.22),
        );
    }

    // train-bpb curve
    let n = snap.bpb_history.len();
    if n >= 2 {
        let mut prev: Option<Vec2> = None;
        for (i, &(_, v)) in snap.bpb_history.iter().enumerate() {
            if !v.is_finite() {
                prev = None;
                continue;
            }
            let cur = map(i as f32, n as f32, v);
            if let Some(p) = prev {
                gizmos.line_2d(p, cur, Color::srgb(0.30, 0.78, 0.50));
            }
            prev = Some(cur);
        }
    }
}

fn refresh_status(live: Res<ThermoLive>, mut q: Query<(&mut Text2d, &mut TextColor), With<ThermoStatus>>) {
    let Some(snap) = live.0.as_ref() else { return };
    let Ok((mut text, mut color)) = q.single_mut() else { return };
    let lead = if snap.finished { "[DONE]" } else { "THERMO" };
    let val = if snap.val_bpb.is_finite() {
        format!("{:.3}", snap.val_bpb)
    } else {
        "—".into()
    };
    text.0 = format!(
        "{lead}  |  {params}  |  step {step}/{total}  |  loss {loss:.3}  |  train_bpb {tb:.3}  |  val_bpb {val}  |  lr {lr:.1e}  |  grad {gn:.2}  |  {el:.0}s",
        params = format_params(snap.params),
        step = snap.step,
        total = snap.total_steps,
        loss = snap.loss,
        tb = snap.train_bpb,
        lr = snap.lr,
        gn = snap.grad_norm,
        el = snap.elapsed_s,
    );
    *color = if snap.finished {
        TextColor(Color::srgb(0.40, 0.82, 0.52))
    } else {
        TextColor(Color::srgb(0.86, 0.90, 0.96))
    };
}

fn refresh_subtitle(live: Res<ThermoLive>, mut q: Query<(&mut Text2d, &mut TextColor), With<ThermoSubtitle>>) {
    let Some(snap) = live.0.as_ref() else { return };
    let Ok((mut text, mut color)) = q.single_mut() else { return };
    let verdict = if snap.best_val.is_finite() && snap.bigram_bpb.is_finite() {
        let d = snap.bigram_bpb - snap.best_val;
        if d > 0.0 {
            format!("best val {:.3}  →  BEATS bigram by {:.3}", snap.best_val, d)
        } else {
            format!("best val {:.3}  (bigram −{:.3})", snap.best_val, -d)
        }
    } else {
        "best val —".into()
    };
    text.0 = format!(
        "grid {h}×{w} · K {k} · dt {dt}  |  bigram baseline {bg:.3} bpb  |  {verdict}",
        h = snap.h, w = snap.w, k = snap.k_steps, dt = snap.dt, bg = snap.bigram_bpb,
    );
    *color = if snap.best_val.is_finite() && snap.best_val < snap.bigram_bpb {
        TextColor(Color::srgb(0.40, 0.82, 0.52))
    } else {
        TextColor(Color::srgb(0.62, 0.70, 0.82))
    };
}

fn refresh_tokens(live: Res<ThermoLive>, mut q: Query<(&mut Text2d, &mut TextColor), With<ThermoTokens>>) {
    let Some(snap) = live.0.as_ref() else { return };
    let Ok((mut text, mut color)) = q.single_mut() else { return };
    let inp = if snap.input_preview.is_empty() {
        "(no tokens yet)".into()
    } else {
        sanitize(&snap.input_preview, 80)
    };
    let pred = if snap.pred_preview.is_empty() {
        "-".into()
    } else {
        sanitize(&snap.pred_preview, 12)
    };
    let tag = match snap.pred_correct {
        Some(true) => "OK",
        Some(false) => "MISS",
        None => "..",
    };
    text.0 = format!("input:  {inp}    ->  pred[{tag}]: {pred}");
    *color = match snap.pred_correct {
        Some(true) => TextColor(Color::srgb(0.40, 0.82, 0.52)),
        Some(false) => TextColor(Color::srgb(0.86, 0.34, 0.30)),
        None => TextColor(Color::srgb(0.74, 0.80, 0.90)),
    };
}

/// One-shot framebuffer screenshot, only when `THERMO_VIZ_SHOT=<path>` is
/// set. Fires once ~4s in (enough for the training thread to populate the
/// field + curve), then never again. Normal `--viz` runs write nothing.
fn auto_screenshot(mut st: ResMut<ShotState>, mut commands: Commands) {
    if st.fired {
        return;
    }
    st.frame += 1;
    if st.frame < 240 {
        return;
    }
    if let Ok(path) = std::env::var("THERMO_VIZ_SHOT") {
        st.fired = true;
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
        info!("thermo viz: screenshot requested");
    }
}

fn sanitize(s: &str, max: usize) -> String {
    let cleaned: String = s.chars().map(|c| if c.is_control() { '.' } else { c }).collect();
    if cleaned.chars().count() > max {
        let mut out: String = cleaned.chars().take(max).collect();
        out.push('…');
        out
    } else {
        cleaned
    }
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M params", n as f64 / 1e6)
    } else if n >= 1000 {
        format!("{:.0}K params", n as f64 / 1e3)
    } else {
        format!("{n} params")
    }
}
