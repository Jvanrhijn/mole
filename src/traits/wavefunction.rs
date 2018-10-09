use ndarray::{Array2};

pub trait WaveFunction {
    fn gradient(&self, configuration: &Array2<f64>) -> f64;
}

