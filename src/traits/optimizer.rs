use ndarray::{Array1};

#[allow(dead_code)]
pub trait Optimizer {
    fn step(&self, parms: &mut Array1<f64>, grads: &Array1<f64>);
}
