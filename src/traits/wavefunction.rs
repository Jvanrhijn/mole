use ndarray::{Array1};

pub trait WaveFunction {
    fn gradient(&self, configuration: &Array1<f64>) -> f64;
}