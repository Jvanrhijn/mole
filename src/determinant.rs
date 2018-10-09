// Standard imports
use std::vec::{Vec};
// Third party imports
use ndarray::{Ix2, Array};
use ndarray_linalg::error::LinalgError;
// First party imports
use traits::function::*;
use math::mat_ops;

pub struct Determinant<T: Function<f64>> {
    orbs: Vec<T>
}

impl<T: Function<f64>> Determinant<T> {
    pub fn new(orbs: Vec<T>) -> Self {
       Determinant{orbs}
    }
}

impl<T: Function<f64, D=Ix2>> Function<f64> for Determinant<T> {

    type D = Ix2;
    type E = LinalgError;

    fn value(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Self::E> {
        let mat_dim = (self.orbs.len() as f64).sqrt().round() as usize;
        let matrix = Array::from_vec(self.orbs.iter().map(|x| {
            x.value(cfg).expect("Failed to calculate determinant term")
        }).collect()).into_shape((mat_dim, mat_dim))?;
        mat_ops::det_abs(&matrix)
    }

}
