use ndarray::{Array1, Array2};

pub trait Optimize {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Array1<f64>;
}
