#![allow(clippy::cast_precision_loss)]

use aurora::encoding::rate_encode_bernoulli;
use aurora::snn_dense::SurrogateSnn;
use aurora::snn_loss::argmax;
use aurora::surrogate::SpikeFn;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SIDE: usize = 4;
const N_INPUT: usize = SIDE * SIDE;
const N_HIDDEN: usize = 32;
const N_CLASSES: usize = 2;
const T_STEPS: usize = 25;
const MAX_RATE: f32 = 0.4;

fn make_horizontal(rng: &mut impl Rng, noise: f32) -> Vec<f32> {
    let mut img = vec![0.0_f32; N_INPUT];
    for c in 0..SIDE {
        img[SIDE + c] = 1.0;
        img[2 * SIDE + c] = 1.0;
    }
    for v in img.iter_mut() {
        *v = (*v + rng.random::<f32>() * noise).clamp(0.0, 1.0);
    }
    img
}

fn make_vertical(rng: &mut impl Rng, noise: f32) -> Vec<f32> {
    let mut img = vec![0.0_f32; N_INPUT];
    for r in 0..SIDE {
        img[r * SIDE + 1] = 1.0;
        img[r * SIDE + 2] = 1.0;
    }
    for v in img.iter_mut() {
        *v = (*v + rng.random::<f32>() * noise).clamp(0.0, 1.0);
    }
    img
}

fn make_sample(class: usize, rng: &mut impl Rng) -> Vec<f32> {
    if class == 0 {
        make_horizontal(rng, 0.15)
    } else {
        make_vertical(rng, 0.15)
    }
}

#[test]
fn surrogate_gradient_classifies_stripes() {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut net = SurrogateSnn::new(
        N_INPUT,
        N_HIDDEN,
        N_CLASSES,
        SpikeFn::default(),
        5.0,
        1.0,
        &mut rng,
    );

    let n_train_per_class = 200;
    let lr = 0.005;
    let mut last_acc = 0.0;
    for epoch in 0..3 {
        let mut correct = 0;
        let mut total = 0;
        for _ in 0..n_train_per_class {
            for class in 0..N_CLASSES {
                let img = make_sample(class, &mut rng);
                let encoded = rate_encode_bernoulli(&img, T_STEPS, MAX_RATE, &mut rng);
                let cache = net.forward(&encoded);
                net.zero_grad();
                let (_loss, pred) = net.backward(&cache, &encoded, class);
                if pred == class {
                    correct += 1;
                }
                total += 1;
                net.step_sgd(lr);
            }
        }
        last_acc = f64::from(correct as u32) / f64::from(total as u32);
        eprintln!("epoch {epoch} train acc {last_acc:.3}");
    }

    let n_test_per_class = 50;
    let mut correct = 0;
    let mut total = 0;
    for _ in 0..n_test_per_class {
        for class in 0..N_CLASSES {
            let img = make_sample(class, &mut rng);
            let encoded = rate_encode_bernoulli(&img, T_STEPS, MAX_RATE, &mut rng);
            let cache = net.forward(&encoded);
            if argmax(&cache.spike_counts) == class {
                correct += 1;
            }
            total += 1;
        }
    }
    let test_acc = f64::from(correct as u32) / f64::from(total as u32);
    eprintln!("test acc {test_acc:.3} (last train acc {last_acc:.3})");
    assert!(
        test_acc >= 0.95,
        "surrogate gradient must reach ≥95% on stripes, got {test_acc:.3}"
    );
}
