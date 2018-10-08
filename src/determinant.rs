// Standard imports
use std::vec::{Vec};
// First party imports
use traits::orbital::*;

pub struct Determinant<T: Orbital> {
    orbs: Vec<T>
}

impl<T> Determinant<T> {
    pub fn new(orbs: Vec<T>) -> Self {
       Determinant{orbs}
    }

    pub fn value(&self, cfg: Array1<f64>) -> f64 {
        // stub TODO write value method for Determinant
        1.0
    }
}

