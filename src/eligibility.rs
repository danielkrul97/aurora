//! Eligibility trace parameters. The trace itself lives on each synapse;
//! these constants govern its decay and the weight-update magnitude.

use crate::types::Time;

#[derive(Copy, Clone, Debug)]
pub struct EligibilityParams {
    pub tau: Time,
    pub learning_rate: f64,
}

impl Default for EligibilityParams {
    fn default() -> Self {
        Self {
            tau: 0.500,
            learning_rate: 1.0,
        }
    }
}
