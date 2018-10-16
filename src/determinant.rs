// Standard imports
use std::vec::Vec;
// Third party imports
use ndarray::{Ix2, Ix1, Array, Array1, Array2, Axis, ArrayBase};
use ndarray_linalg::{Inverse, error, error::{LinalgError, LinalgError::Shape}, qr::QRSquare};
// First party imports
use traits::function::*;
use traits::wavefunction::WaveFunction;
use math::mat_ops;

pub struct Determinant<T: Function<f64, D=Ix1>> {
    orbs: Vec<T>,
    matrix: Array2<f64>
}

impl<T: Function<f64, D=Ix1>> Determinant<T> {

    pub fn new(orbs: Vec<T>) -> Self {
        let mat_dim = orbs.len();
        Determinant{orbs, matrix: Array::<f64, Ix2>::eye(mat_dim)}
    }

    pub fn build_matrix(&mut self, cfg: &Array2<f64>) {
        let mat_dim = self.orbs.len();
        // build the Slater determinantal matrix
        for i in 0..mat_dim {
            for j in 0..mat_dim {
                let slice = cfg.slice(s![i, ..]);
                let pos = array![slice[0], slice[1], slice[2]];
                self.matrix[[i, j]] = self.orbs[j].value(&pos).unwrap();
            }
        }
    }

}

impl<T> Function<f64> for Determinant<T> where T: Function<f64, D=Ix1> {

    type E = LinalgError;
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Self::E> {
        /*match &self.matrix {
            Some(matrix) => mat_ops::det_abs(&matrix),
            None => Err(error::LinalgError::from(0))
        }*/
        mat_ops::det_abs(&self.matrix)
    }

}

impl<T> WaveFunction for Determinant<T>
where T: Function<f64, D=Ix1> + WaveFunction<D=Ix1>
{
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Array2<f64> {
        // TODO implement
        let shape = cfg.shape();
        Array2::<f64>::ones((shape[0], shape[1]))
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> f64 {
        // TODO implement more efficienctly
        let old_method: f64 = self.orbs.iter().zip(cfg.axis_iter(Axis(0))).map(|(x, y)| x.laplacian(&array![y[0], y[1], y[2]])).sum();
        let mat_dim = self.orbs.len();
        let det = mat_ops::det_abs(&self.matrix).unwrap();
        let mat_inv = self.matrix.inv().unwrap();
        let mut result = 0.;
        for i in 0..mat_dim {
            let ri = array![cfg[[i, 0]], cfg[[i, 1]], cfg[[i, 2]]];
            for j in 0..mat_dim {
                result += self.orbs[j].laplacian(&ri)*mat_inv[[j, i]];
            }
        }
        result*det
    }
}
