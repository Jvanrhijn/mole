use ndarray::{Array1};

pub trait WaveFunction {
    fn value(&self, configuration: &Array1<f64>) -> f64;
}