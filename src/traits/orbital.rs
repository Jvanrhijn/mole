// Third party import
use ndarray::{Array1};

pub trait Orbital {
    fn value(&self, pos: Array1<f64>) -> f64;
}