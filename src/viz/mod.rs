//! Real-time Bevy visualization of the Aurora training run.
//!
//! [`run_window`] must be called on the OS main thread (winit requirement)
//! and blocks until the window is closed.

pub mod components;
pub mod scene;
pub mod snapshot;
pub mod systems;
pub mod ui;

pub use snapshot::{
    health_from_norm, l2_norm, max_abs, new_snapshot, LayerKind, LayerStats, SnapshotArc,
    TrainingSnapshot,
};

use bevy::log::LogPlugin;
use bevy::prelude::*;

use components::{LastStepTick, SceneLayout, SnapshotHandle};
use systems::LiveSnapshot;

/// `n_layer` counts transformer blocks only; the pipeline view adds
/// embed + lm-head stages on either side.
pub fn run_window(snapshot: SnapshotArc, n_layer: usize) {
    let total_stages = n_layer + 2;

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Aurora — live training".into(),
                        resolution: (1400u32, 900u32).into(),
                        ..default()
                    }),
                    ..default()
                })
                // Silence the harmless surface-reconfigure warning Bevy emits
                // on the first frame while the compositor settles DPI/scale.
                .set(LogPlugin {
                    filter: "wgpu=error,naga=warn,bevy_render::view::window=error".into(),
                    level: bevy::log::Level::INFO,
                    ..default()
                }),
        )
        .insert_resource(ClearColor(Color::srgb(0.97, 0.97, 0.96)))
        .insert_resource(SnapshotHandle(snapshot))
        .insert_resource(LiveSnapshot::default())
        .insert_resource(LastStepTick::default())
        .insert_resource(SceneLayout {
            n_layer,
            total_stages,
        })
        .add_systems(Startup, scene::setup)
        .add_systems(
            Update,
            (
                systems::poll_snapshot,
                systems::trigger_pulses,
                systems::decay_pulses,
                systems::update_layer_colors,
                systems::pulse_layer_scale,
                systems::update_attn_bars,
                systems::update_ffn_bars,
                systems::spawn_particles,
                systems::advance_particles,
                systems::draw_loss_chart,
                ui::refresh_status_bar,
                ui::refresh_tokens_strip,
            )
                .chain(),
        )
        .run();
}
