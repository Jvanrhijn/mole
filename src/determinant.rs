// Standard imports
use std::vec::Vec;
// Third party imports
use ndarray::{Ix2, Ix1, Array, Array1, Array2};
use ndarray_linalg::error::LinalgError;
// First party imports
use traits::function::*;
use math::mat_ops;

pub struct Determinant<T: Function<f64, D=Ix1>> {
    orbs: Vec<T>
}

impl<T: Function<f64, D=Ix1>> Determinant<T> {
    pub fn new(orbs: Vec<T>) -> Self {
       Determinant{orbs}
    }
}

impl<T> Function<f64> for Determinant<T> where T: Function<f64, D=Ix1> {

    type E = LinalgError;
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Self::E> {
        let mat_dim = self.orbs.len();
        // build the Slater determinantal matrix
        let mut matrix = Array2::<f64>::zeros((mat_dim, mat_dim));
        for i in 0..mat_dim {
            for j in 0..mat_dim {
                let slice = cfg.slice(s![j, ..]);
                let pos = Array1::<f64>::from_vec(vec![slice[0], slice[1], slice[2]]);
                matrix[[i, j]] = self.orbs[i].value(&pos).unwrap();
            }
        }
        mat_ops::det_abs(&matrix)
    }

}
