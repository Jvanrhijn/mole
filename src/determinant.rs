// Standard imports
use std::vec::{Vec};
// Third party imports
use ndarray::{Ix2, Array};
// First party imports
use traits::orbital::*;

pub struct Determinant<T: Orbital> {
    orbs: Vec<T>
}

impl<T> Determinant<T> {
    pub fn new(orbs: Vec<T>) -> Self {
       Determinant{orbs}
    }
}

impl Function<T> for Determinant<T> {

    type D = Ix2;

    fn value(&self, cfg: &Array<T, &Self::D>) -> f64 {
        // stub
        1.0
    }

}
