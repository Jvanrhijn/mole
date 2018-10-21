// Third party imports
use ndarray::{Array1};

/// Interface for implementing optimization algorithms.
pub trait Optimizer {
    fn step(&self, parms: &mut Array1<f64>, grads: &Array1<f64>);
}
