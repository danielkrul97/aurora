#![allow(clippy::cast_precision_loss)]

//! Class 1 fires for the first KEY_LEN steps then goes silent — only a
//! recurrent net can carry that pulse to the output through the silent tail.

use aurora::snn_loss::argmax;
use aurora::snn_recurrent::RecurrentSurrogateSnn;
use aurora::surrogate::SpikeFn;
use rand::rngs::StdRng;
use rand::SeedableRng;

const T: usize = 8;
const KEY_LEN: usize = 3;
const N_INPUT: usize = 1;
const N_HIDDEN: usize = 24;
const N_CLASSES: usize = 2;

fn make_sample(class: usize) -> Vec<Vec<f32>> {
    let mut seq = vec![vec![0.0_f32; N_INPUT]; T];
    if class == 1 {
        for step in seq.iter_mut().take(KEY_LEN) {
            step[0] = 1.0;
        }
    }
    seq
}

#[test]
fn recurrent_learns_two_class_pulse() {
    let mut rng = StdRng::seed_from_u64(0xBEEF_1234);
    let mut net = RecurrentSurrogateSnn::new(
        N_INPUT,
        N_HIDDEN,
        N_CLASSES,
        SpikeFn::default(),
        5.0,
        1.0,
        &mut rng,
    );

    let n_train_per_class = 100;
    let lr = 0.01;
    for epoch in 0..8 {
        let mut correct = 0;
        let mut total = 0;
        for _ in 0..n_train_per_class {
            for class in 0..N_CLASSES {
                let seq = make_sample(class);
                let cache = net.forward(&seq);
                net.zero_grad();
                let (_loss, pred) = net.backward(&cache, &seq, class);
                if pred == class {
                    correct += 1;
                }
                total += 1;
                net.clip_grads(1.0);
                net.step_sgd(lr);
            }
        }
        let acc = correct as f64 / total as f64;
        eprintln!("epoch {epoch} train acc {acc:.3}");
    }

    let n_test_per_class = 50;
    let mut correct = 0;
    let mut total = 0;
    for _ in 0..n_test_per_class {
        for class in 0..N_CLASSES {
            let seq = make_sample(class);
            let cache = net.forward(&seq);
            if argmax(&cache.spike_counts) == class {
                correct += 1;
            }
            total += 1;
        }
    }
    let acc = correct as f64 / total as f64;
    eprintln!("test acc {acc:.3}");
    assert!(
        acc >= 0.90,
        "recurrent SNN must learn 2-class pulse task to ≥90%, got {acc:.3}"
    );

    let w_rec_norm: f32 = net.w_rec.iter().map(|w| w * w).sum::<f32>().sqrt();
    eprintln!("‖W_rec‖_F = {w_rec_norm:.3}");
    assert!(
        w_rec_norm > 0.001,
        "W_rec must have non-trivial values after training (got norm {w_rec_norm:.6})"
    );
}
