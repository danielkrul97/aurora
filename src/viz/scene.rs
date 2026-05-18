//! Bevy startup system: spawns the static visualization scene.
//!
//! Uses `Mesh2d` + `MeshMaterial2d<ColorMaterial>` for colored shapes —
//! `Sprite::from_color` would render nothing because it leaves the
//! image handle empty.
//!
//! Layout (1400 × 900 default window, world coords centered on (0,0)):
//!
//! ```text
//!   y=+450 ┌─────────────── status bar (UI, 80 px) ───────────────┐
//!          ├──────────────────────────────────────────────────────┤
//!          │   pipeline area  (world coords y ∈ [-370, +370])     │
//!          │                                                      │
//!   y=-450 ├─────────────── tokens strip (UI, 80 px) ─────────────┤
//!          └──────────────────────────────────────────────────────┘
//! ```

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::sprite_render::{ColorMaterial, MeshMaterial2d};

use super::components::{
    AttnBar, EdgeFrom, FfnBar, LayerBox, LayerLabel, LossChart, PulseState, SceneLayout,
    StatusBar, TokensStrip, VizAssets,
};
use super::snapshot::LayerKind;

const STATUS_BAR_HEIGHT: f32 = 80.0;
const TOKENS_STRIP_HEIGHT: f32 = 80.0;
const PIPELINE_TOP_PADDING: f32 = 24.0;
const PIPELINE_BOTTOM_PADDING: f32 = 24.0;

const BOX_BG: Color = Color::srgb(0.84, 0.88, 0.92);
const BOX_BORDER: Color = Color::srgb(0.40, 0.46, 0.56);
const EDGE_COLOR: Color = Color::srgb(0.55, 0.62, 0.72);
const LABEL_COLOR: Color = Color::srgb(0.10, 0.14, 0.20);
const BAR_BG: Color = Color::srgb(0.78, 0.82, 0.86);
const ATTN_COLOR: Color = Color::srgb(0.15, 0.40, 0.72);
const FFN_COLOR: Color = Color::srgb(0.80, 0.32, 0.10);
const PARTICLE_COLOR: Color = Color::srgb(0.12, 0.40, 0.70);
const PANEL_BG: Color = Color::srgb(0.96, 0.96, 0.94);
const BAND_BG: Color = Color::srgb(0.91, 0.93, 0.96);
const BAND_BORDER: Color = Color::srgb(0.50, 0.56, 0.66);

#[derive(Debug, Clone, Copy)]
struct StageGeom {
    stage: usize,
    kind: LayerKind,
    center_x: f32,
    center_y: f32,
    width: f32,
    height: f32,
}

pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    layout: Res<SceneLayout>,
    windows: Query<&Window>,
) {
    commands.spawn(Camera2d);

    let unit_rect = meshes.add(Rectangle::new(1.0, 1.0));

    let edge_mat = materials.add(ColorMaterial::from_color(EDGE_COLOR));
    let border_mat = materials.add(ColorMaterial::from_color(BOX_BORDER));
    let bar_bg_mat = materials.add(ColorMaterial::from_color(BAR_BG));
    let attn_mat = materials.add(ColorMaterial::from_color(ATTN_COLOR));
    let ffn_mat = materials.add(ColorMaterial::from_color(FFN_COLOR));
    let panel_mat = materials.add(ColorMaterial::from_color(PANEL_BG));
    let particle_mat = materials.add(ColorMaterial::from_color(PARTICLE_COLOR));

    let assets = VizAssets {
        unit_rect: unit_rect.clone(),
        edge_mat: edge_mat.clone(),
        border_mat: border_mat.clone(),
        bar_bg_mat: bar_bg_mat.clone(),
        attn_mat: attn_mat.clone(),
        ffn_mat: ffn_mat.clone(),
        panel_mat: panel_mat.clone(),
        particle_mat: particle_mat.clone(),
    };
    commands.insert_resource(assets);

    let (win_w, win_h) = windows
        .single()
        .map(|w| (w.resolution.width(), w.resolution.height()))
        .unwrap_or((1400.0, 900.0));

    let pipeline_top = win_h * 0.5 - STATUS_BAR_HEIGHT - PIPELINE_TOP_PADDING;
    let pipeline_bottom = -win_h * 0.5 + TOKENS_STRIP_HEIGHT + PIPELINE_BOTTOM_PADDING;
    let pipeline_h = pipeline_top - pipeline_bottom;

    let pipeline_center_x = -win_w * 0.12;
    let box_w = (win_w * 0.30).clamp(220.0, 380.0);

    let n_stages = layout.total_stages.max(1) as f32;
    let per_stage = pipeline_h / n_stages;
    let box_h = (per_stage * 0.72).clamp(28.0, 86.0);

    let geoms: Vec<StageGeom> = (0..layout.total_stages)
        .map(|i| {
            let kind = match i {
                0 => LayerKind::Embed,
                k if k == layout.total_stages - 1 => LayerKind::LmHead,
                _ => LayerKind::Block,
            };
            let center_y = pipeline_top - per_stage * (i as f32 + 0.5);
            StageGeom {
                stage: i,
                kind,
                center_x: pipeline_center_x,
                center_y,
                width: box_w,
                height: box_h,
            }
        })
        .collect();

    spawn_edges(&mut commands, &geoms, &unit_rect, &edge_mat);
    for g in &geoms {
        spawn_stage(&mut commands, &mut materials, g, &unit_rect, &border_mat, &bar_bg_mat, &attn_mat, &ffn_mat);
    }

    spawn_status_bar(&mut commands, &mut materials, &unit_rect, win_w, win_h);
    spawn_tokens_strip(&mut commands, &mut materials, &unit_rect, win_w, win_h);
    spawn_loss_chart_panel(&mut commands, &unit_rect, &panel_mat, win_w, win_h);
}

fn rect_transform(center_x: f32, center_y: f32, w: f32, h: f32, z: f32) -> Transform {
    Transform {
        translation: Vec3::new(center_x, center_y, z),
        scale: Vec3::new(w, h, 1.0),
        ..default()
    }
}

fn spawn_edges(
    commands: &mut Commands,
    geoms: &[StageGeom],
    unit_rect: &Handle<Mesh>,
    edge_mat: &Handle<ColorMaterial>,
) {
    for pair in geoms.windows(2) {
        let a = pair[0];
        let b = pair[1];
        let top_y = a.center_y - a.height * 0.5;
        let bot_y = b.center_y + b.height * 0.5;
        let center_y = (top_y + bot_y) * 0.5;
        let length = top_y - bot_y;
        if length <= 0.0 {
            continue;
        }
        let start = Vec2::new(a.center_x, top_y);
        let end = Vec2::new(a.center_x, bot_y);
        commands.spawn((
            Mesh2d(unit_rect.clone()),
            MeshMaterial2d(edge_mat.clone()),
            rect_transform(a.center_x, center_y, 3.0, length, -1.0),
            EdgeFrom {
                from_stage: a.stage,
                start_pos: start,
                end_pos: end,
                last_emit_s: 0.0,
            },
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_stage(
    commands: &mut Commands,
    materials: &mut Assets<ColorMaterial>,
    g: &StageGeom,
    unit_rect: &Handle<Mesh>,
    border_mat: &Handle<ColorMaterial>,
    bar_bg_mat: &Handle<ColorMaterial>,
    attn_mat: &Handle<ColorMaterial>,
    ffn_mat: &Handle<ColorMaterial>,
) {
    commands.spawn((
        Mesh2d(unit_rect.clone()),
        MeshMaterial2d(border_mat.clone()),
        rect_transform(g.center_x, g.center_y, g.width + 4.0, g.height + 4.0, 0.0),
    ));

    // Own material per stage so health colours can diverge.
    let fill_mat = materials.add(ColorMaterial::from_color(BOX_BG));
    commands.spawn((
        Mesh2d(unit_rect.clone()),
        MeshMaterial2d(fill_mat.clone()),
        rect_transform(g.center_x, g.center_y, g.width, g.height, 0.1),
        LayerBox {
            stage: g.stage,
            material: fill_mat,
            base_w: g.width,
            base_h: g.height,
            base_x: g.center_x,
            base_y: g.center_y,
        },
        PulseState::default(),
    ));

    let label = match g.kind {
        LayerKind::Embed => "EMBED".to_string(),
        LayerKind::Block => format!("BLOCK {}", g.stage),
        LayerKind::LmHead => "LM HEAD".to_string(),
    };
    let font_size = (g.height * 0.34).clamp(14.0, 24.0);
    commands.spawn((
        Text2d::new(label),
        TextFont {
            font_size,
            ..default()
        },
        TextColor(LABEL_COLOR),
        Transform::from_xyz(g.center_x - g.width * 0.32, g.center_y, 0.2),
        LayerLabel { stage: g.stage },
    ));

    if matches!(g.kind, LayerKind::Block) {
        spawn_inner_bars(commands, g, unit_rect, bar_bg_mat, attn_mat, ffn_mat);
    }
}

fn spawn_inner_bars(
    commands: &mut Commands,
    g: &StageGeom,
    unit_rect: &Handle<Mesh>,
    bar_bg_mat: &Handle<ColorMaterial>,
    attn_mat: &Handle<ColorMaterial>,
    ffn_mat: &Handle<ColorMaterial>,
) {
    let bar_w = g.width * 0.35;
    let bar_h = (g.height * 0.18).clamp(5.0, 12.0);
    let bar_center_x = g.center_x + g.width * 0.20;
    let bar_left = bar_center_x - bar_w * 0.5;
    let attn_y = g.center_y + bar_h * 0.9;
    let ffn_y = g.center_y - bar_h * 0.9;

    commands.spawn((
        Mesh2d(unit_rect.clone()),
        MeshMaterial2d(bar_bg_mat.clone()),
        rect_transform(bar_center_x, attn_y, bar_w, bar_h, 0.2),
    ));
    commands.spawn((
        Mesh2d(unit_rect.clone()),
        MeshMaterial2d(bar_bg_mat.clone()),
        rect_transform(bar_center_x, ffn_y, bar_w, bar_h, 0.2),
    ));

    // Live bars: scale.x is the bar width and translation.x = bar_left + scale.x/2
    // so the left edge stays anchored regardless of growth.
    let initial_w = 1.0;
    commands.spawn((
        Mesh2d(unit_rect.clone()),
        MeshMaterial2d(attn_mat.clone()),
        rect_transform(bar_left + initial_w * 0.5, attn_y, initial_w, bar_h, 0.3),
        AttnBar {
            stage: g.stage,
            max_w: bar_w,
            anchor_x: bar_left,
            height: bar_h,
        },
    ));
    commands.spawn((
        Mesh2d(unit_rect.clone()),
        MeshMaterial2d(ffn_mat.clone()),
        rect_transform(bar_left + initial_w * 0.5, ffn_y, initial_w, bar_h, 0.3),
        FfnBar {
            stage: g.stage,
            max_w: bar_w,
            anchor_x: bar_left,
            height: bar_h,
        },
    ));

    let font_size = (bar_h * 1.2).clamp(8.0, 12.0);
    commands.spawn((
        Text2d::new("attn"),
        TextFont {
            font_size,
            ..default()
        },
        TextColor(Color::srgb(0.10, 0.28, 0.52)),
        Transform::from_xyz(bar_left - 22.0, attn_y, 0.3),
    ));
    commands.spawn((
        Text2d::new("ffn"),
        TextFont {
            font_size,
            ..default()
        },
        TextColor(Color::srgb(0.55, 0.22, 0.08)),
        Transform::from_xyz(bar_left - 22.0, ffn_y, 0.3),
    ));
}

fn spawn_status_bar(
    commands: &mut Commands,
    materials: &mut Assets<ColorMaterial>,
    unit_rect: &Handle<Mesh>,
    win_w: f32,
    win_h: f32,
) {
    let band_h = STATUS_BAR_HEIGHT;
    let band_y = win_h * 0.5 - band_h * 0.5;
    let band_mat = materials.add(ColorMaterial::from_color(BAND_BG));
    let border_mat = materials.add(ColorMaterial::from_color(BAND_BORDER));
    commands.spawn((
        Mesh2d(unit_rect.clone()),
        MeshMaterial2d(band_mat),
        rect_transform(0.0, band_y, win_w, band_h, 5.0),
    ));
    commands.spawn((
        Mesh2d(unit_rect.clone()),
        MeshMaterial2d(border_mat),
        rect_transform(0.0, band_y - band_h * 0.5 + 1.0, win_w, 2.0, 5.1),
    ));
    commands.spawn((
        Text2d::new("AURORA  |  waiting for first step..."),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::srgb(0.12, 0.16, 0.22)),
        TextLayout::new_with_justify(Justify::Left),
        Transform::from_xyz(-win_w * 0.5 + 24.0, band_y, 5.2),
        Anchor::CENTER_LEFT,
        StatusBar,
    ));
}

fn spawn_tokens_strip(
    commands: &mut Commands,
    materials: &mut Assets<ColorMaterial>,
    unit_rect: &Handle<Mesh>,
    win_w: f32,
    win_h: f32,
) {
    let band_h = TOKENS_STRIP_HEIGHT;
    let band_y = -win_h * 0.5 + band_h * 0.5;
    let band_mat = materials.add(ColorMaterial::from_color(BAND_BG));
    let border_mat = materials.add(ColorMaterial::from_color(BAND_BORDER));
    commands.spawn((
        Mesh2d(unit_rect.clone()),
        MeshMaterial2d(band_mat),
        rect_transform(0.0, band_y, win_w, band_h, 5.0),
    ));
    commands.spawn((
        Mesh2d(unit_rect.clone()),
        MeshMaterial2d(border_mat),
        rect_transform(0.0, band_y + band_h * 0.5 - 1.0, win_w, 2.0, 5.1),
    ));
    commands.spawn((
        Text2d::new("input:  (no tokens yet)        ->  pred: -"),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgb(0.20, 0.25, 0.32)),
        TextLayout::new_with_justify(Justify::Left),
        Transform::from_xyz(-win_w * 0.5 + 24.0, band_y, 5.2),
        Anchor::CENTER_LEFT,
        TokensStrip,
    ));
}

fn spawn_loss_chart_panel(
    commands: &mut Commands,
    unit_rect: &Handle<Mesh>,
    panel_mat: &Handle<ColorMaterial>,
    win_w: f32,
    win_h: f32,
) {
    let panel_w = win_w * 0.28;
    let panel_h = (win_h * 0.30).min(260.0);
    let center_x = win_w * 0.5 - panel_w * 0.5 - 24.0;
    let center_y = win_h * 0.5 - STATUS_BAR_HEIGHT - panel_h * 0.5 - 24.0;
    let center = Vec2::new(center_x, center_y);
    let size = Vec2::new(panel_w, panel_h);
    commands.spawn((
        Mesh2d(unit_rect.clone()),
        MeshMaterial2d(panel_mat.clone()),
        rect_transform(center_x, center_y, panel_w, panel_h, -0.5),
        LossChart { center, size },
    ));

    let label_y = center_y + panel_h * 0.5 - 18.0;
    commands.spawn((
        Text2d::new("LOSS"),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor(Color::srgb(0.20, 0.28, 0.38)),
        Transform::from_xyz(center_x - panel_w * 0.5 + 22.0, label_y, 0.0),
    ));
}
