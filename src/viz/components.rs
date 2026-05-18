//! Bevy marker components and resource handles for the viz scene.

use bevy::prelude::*;
use bevy::sprite_render::ColorMaterial;

use super::snapshot::SnapshotArc;

#[derive(Resource)]
pub struct SnapshotHandle(pub SnapshotArc);

#[derive(Resource, Debug, Clone, Copy)]
pub struct SceneLayout {
    pub n_layer: usize,
    pub total_stages: usize,
}

#[derive(Component, Debug, Clone)]
pub struct LayerBox {
    pub stage: usize,
    pub material: Handle<ColorMaterial>,
    pub base_w: f32,
    pub base_h: f32,
    pub base_x: f32,
    pub base_y: f32,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct LayerLabel {
    pub stage: usize,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct AttnBar {
    pub stage: usize,
    pub max_w: f32,
    pub anchor_x: f32,
    pub height: f32,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct FfnBar {
    pub stage: usize,
    pub max_w: f32,
    pub anchor_x: f32,
    pub height: f32,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct EdgeFrom {
    pub from_stage: usize,
    pub start_pos: Vec2,
    pub end_pos: Vec2,
    pub last_emit_s: f32,
}

#[derive(Component)]
pub struct StatusBar;

#[derive(Component)]
pub struct TokensStrip;

#[derive(Component, Debug, Clone, Copy)]
pub struct LossChart {
    pub center: Vec2,
    pub size: Vec2,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct Particle {
    pub progress: f32,
    pub speed: f32,
    pub start: Vec2,
    pub end: Vec2,
}

#[derive(Component, Debug, Default)]
pub struct PulseState {
    pub remaining_s: f32,
    pub last_step_seen: u64,
}

#[derive(Resource, Debug, Default)]
pub struct LastStepTick(pub u64);

#[derive(Resource, Debug, Clone)]
pub struct VizAssets {
    pub unit_rect: Handle<Mesh>,
    pub edge_mat: Handle<ColorMaterial>,
    pub border_mat: Handle<ColorMaterial>,
    pub bar_bg_mat: Handle<ColorMaterial>,
    pub attn_mat: Handle<ColorMaterial>,
    pub ffn_mat: Handle<ColorMaterial>,
    pub panel_mat: Handle<ColorMaterial>,
    pub particle_mat: Handle<ColorMaterial>,
}
