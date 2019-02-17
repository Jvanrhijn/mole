// Standard imports
use std::vec::Vec;
// Third party imports
use ndarray::{Array, Array1, Ix1};
// First party imports
use crate::traits::{Function, Differentiate};
use crate::error::Error;

/// Parametrized orbital as a linear combination of basis functions:
/// $\phi(x) = \sum_{i=1}^{N_{\text{basis}}} \xi_i(x)$.
pub struct Orbital<'a, T: 'a>
    where T: ?Sized + Fn(&Array1<f64>) -> (f64, f64)
{
    parms: Array1<f64>,
    basis_set: &'a Vec<Box<T>>
}

impl<'a, T> Orbital<'a, T>
    where T: ?Sized + Fn(&Array1<f64>) -> (f64, f64)
{
    pub fn new(parms: Array1<f64>, basis_set: &'a Vec<Box<T>>) -> Self {
        Self{parms, basis_set}
    }
}

impl<'a, T> Function<f64> for Orbital<'a, T>
    where T: ?Sized + Fn(&Array1<f64>) -> (f64, f64) {

    type D = Ix1;

    fn value(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        let basis_vals = self.basis_set.iter().map(|x| x(cfg).0).collect();
        Ok((Array1::from_vec(basis_vals) * &self.parms).scalar_sum())
    }
}

impl<'a, T> Differentiate for Orbital<'a, T>
    where T: ?Sized + Fn(&Array1<f64>) -> (f64, f64) {

    type D = Ix1;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Array<f64, Self::D> {
        Array1::<f64>::ones(cfg.len())
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        Ok(self.parms.iter().zip(self.basis_set).map(|(x, y)| x*y(cfg).1).sum())
    }

}
