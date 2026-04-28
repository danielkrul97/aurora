#![allow(clippy::suboptimal_flops)]

//! Spike function with fast-sigmoid surrogate gradient
//! (Neftci et al. 2019, Zenke & Vogels 2021).

#[derive(Copy, Clone, Debug)]
pub struct SpikeFn {
    pub threshold: f32,
    pub alpha: f32,
}

impl Default for SpikeFn {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            alpha: 4.0,
        }
    }
}

impl SpikeFn {
    #[must_use]
    pub fn forward(&self, v: f32) -> f32 {
        if v >= self.threshold {
            1.0
        } else {
            0.0
        }
    }

    #[must_use]
    pub fn backward(&self, v: f32) -> f32 {
        let x = v - self.threshold;
        let denom = 1.0 + self.alpha * x.abs();
        1.0 / (denom * denom)
    }
}
