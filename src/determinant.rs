// Standard imports
use std::vec::Vec;
// Third party imports
use ndarray::{Ix2, Ix1, Array, Array1, Array2, Axis};
use ndarray_linalg::error::LinalgError;
// First party imports
use traits::function::*;
use traits::wavefunction::WaveFunction;
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
                let slice = cfg.slice(s![i, ..]);
                let pos = array![slice[0], slice[1], slice[2]];
                matrix[[i, j]] = self.orbs[j].value(&pos).unwrap();
            }
        }
        mat_ops::det_abs(&matrix)
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
        // TODO implement
        // fake implementation:
        self.orbs.iter().zip(cfg.axis_iter(Axis(0))).map(|(x, y)| x.laplacian(&array![y[0], y[1], y[2]])).sum()
    }
}
