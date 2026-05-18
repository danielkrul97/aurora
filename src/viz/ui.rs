//! Status bar (top) and tokens strip (bottom). World-space `Text2d` to
//! sidestep Bevy's flex layout quirks for the fixed-position bands.

use bevy::prelude::*;

use super::components::{StatusBar, TokensStrip};
use super::systems::LiveSnapshot;

pub fn refresh_status_bar(
    live: Res<LiveSnapshot>,
    mut q: Query<(&mut Text2d, &mut TextColor), With<StatusBar>>,
) {
    let Some(snap) = live.0.as_ref() else {
        return;
    };
    let Ok((mut text, mut color)) = q.single_mut() else {
        return;
    };
    let lead = if snap.finished { "[DONE]" } else { "AURORA" };
    let line = format!(
        "{lead}  |  {backend}  |  {params}  |  epoch {epoch}/{total_epochs}  |  step {step}  |  loss {loss:.3}  |  bpb {bpb:.3}  |  lr {lr:.2e}  |  grad {gn:.2}  |  {elapsed:.1}s",
        backend = snap.backend,
        params = format_n(snap.params),
        epoch = snap.epoch + 1,
        total_epochs = snap.total_epochs,
        step = snap.step,
        loss = snap.loss,
        bpb = snap.bpb,
        lr = snap.lr,
        gn = snap.grad_norm,
        elapsed = snap.elapsed_s,
    );
    text.0 = line;
    *color = if snap.finished {
        TextColor(Color::srgb(0.14, 0.50, 0.28))
    } else {
        TextColor(Color::srgb(0.12, 0.16, 0.22))
    };
}

pub fn refresh_tokens_strip(
    live: Res<LiveSnapshot>,
    mut q: Query<(&mut Text2d, &mut TextColor), With<TokensStrip>>,
) {
    let Some(snap) = live.0.as_ref() else {
        return;
    };
    let Ok((mut text, mut color)) = q.single_mut() else {
        return;
    };

    let input_preview = if snap.input_preview.is_empty() {
        "(no tokens yet)".to_string()
    } else {
        sanitize(&snap.input_preview, 90)
    };
    let pred_preview = if snap.pred_preview.is_empty() {
        "-".to_string()
    } else {
        sanitize(&snap.pred_preview, 16)
    };
    let arrow = match snap.pred_correct {
        Some(true) => "OK",
        Some(false) => "MISS",
        None => "..",
    };
    text.0 = format!("input:  {input_preview}   ->  pred[{arrow}]: {pred_preview}");

    *color = match snap.pred_correct {
        Some(true) => TextColor(Color::srgb(0.14, 0.50, 0.28)),
        Some(false) => TextColor(Color::srgb(0.78, 0.18, 0.18)),
        None => TextColor(Color::srgb(0.20, 0.25, 0.32)),
    };
}

fn sanitize(s: &str, max_chars: usize) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_control() { '.' } else { c })
        .collect();
    if cleaned.chars().count() > max_chars {
        let mut out: String = cleaned.chars().take(max_chars).collect();
        out.push_str("...");
        out
    } else {
        cleaned
    }
}

fn format_n(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M params", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K params", n as f64 / 1e3)
    } else {
        format!("{n} params")
    }
}
