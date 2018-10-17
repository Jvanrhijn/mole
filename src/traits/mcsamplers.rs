use std::vec::Vec;

use error::{Error};

pub trait MonteCarloSampler {
    fn sample(&self) -> Vec<f64>;
}
