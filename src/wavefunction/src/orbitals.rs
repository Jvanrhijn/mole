// Third party imports
use ndarray::{Array, Array2, Ix1};
// First party imports
use crate::error::Error;
use crate::traits::{Differentiate, Function};
use basis::BasisSet;

/// Parametrized orbital as a linear combination of basis functions:
/// $\phi(x) = \sum_{i=1}^{N_{\text{basis}}} \xi_i(x)$.
pub struct Orbital<T>
where
    T: BasisSet, //?Sized + Fn(&Array1<f64>) -> Vgl
{
    parms: Array2<f64>,
    basis_set: T,
}

impl<T> Orbital<T>
where
    T: BasisSet, //?Sized + Fn(&Array1<f64>) -> Vgl
{
    pub fn new(parms: Array2<f64>, basis_set: T) -> Self {
        Self { parms, basis_set }
    }
}

impl<T> Function<f64> for Orbital<T>
where
    T: BasisSet, //?Sized + Fn(&Array1<f64>) -> Vgl
{
    type D = Ix1;

    fn value(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        Ok(self.basis_set.linear_combination(cfg, &self.parms).0)
    }
}

impl<T> Differentiate for Orbital<T>
where
    T: BasisSet, //?Sized + Fn(&Array1<f64>) -> Vgl
{
    type D = Ix1;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Result<Array<f64, Self::D>, Error> {
        Ok(self.basis_set.linear_combination(cfg, &self.parms).1)
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        Ok(self.basis_set.linear_combination(cfg, &self.parms).2)
    }
}
