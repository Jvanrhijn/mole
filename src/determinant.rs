// Standard imports
use std::vec::Vec;
use std::result::Result;
// Third party imports
use ndarray::{Ix2, Ix1, Array, Array2};
use ndarray_linalg::{solve::Determinant, Inverse};
// First party imports
use traits::function::*;
use traits::wavefunction::WaveFunction;
use traits::differentiate::Differentiate;
use error::{Error};

pub struct Slater<T: Function<f64, D=Ix1>> {
    orbs: Vec<T>,
    matrix: Array2<f64>,
}

#[allow(dead_code)]
impl<T: Function<f64, D=Ix1>> Slater<T> {

    pub fn new(orbs: Vec<T>) -> Self {
        let mat_dim = orbs.len();
        Self{orbs, matrix: Array::<f64, Ix2>::eye(mat_dim)}
    }

    fn build_matrix(&self, cfg: &Array2<f64>) -> Result<Array2<f64>, Error> {
        let mat_dim = self.orbs.len();
        let mut matrix = Array2::<f64>::zeros((mat_dim, mat_dim));
        for i in 0..mat_dim {
            for j in 0..mat_dim {
                let slice = cfg.slice(s![i, ..]);
                let pos = array![slice[0], slice[1], slice[2]];
                matrix[[i, j]] = self.orbs[j].value(&pos)?;
            }
        }
        Ok(matrix)
    }

}

impl<T> Function<f64> for Slater<T> where T: Function<f64, D=Ix1> {

    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        let matrix = self.build_matrix(cfg)?;
        Ok(matrix.det()?)
    }
}

impl<T> Differentiate for Slater<T>
where T: Function<f64, D=Ix1> + Differentiate<D=Ix1>
{
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Array2<f64> {
        // TODO implement
        let shape = cfg.shape();
        Array2::<f64>::ones((shape[0], shape[1]))
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        // TODO implement more efficienctly
        let mat_dim = self.orbs.len();
        let matrix = self.build_matrix(cfg)?;
        let det = matrix.det()?;
        let mat_inv = matrix.inv()?;
        let mut result = 0.;
        for i in 0..mat_dim {
            let ri = array![cfg[[i, 0]], cfg[[i, 1]], cfg[[i, 2]]];
            for j in 0..mat_dim {
                result += self.orbs[j].laplacian(&ri)?*mat_inv[[j, i]];
            }
        }
        Ok(result*det)
    }

}

impl<T> WaveFunction for Slater<T>
where T: Function<f64, D=Ix1> + Differentiate<D=Ix1>
{
    fn num_electrons(&self) -> usize {
        self.orbs.len()
    }
}
