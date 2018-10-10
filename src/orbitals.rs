// Standard imports
use std::vec::Vec;
// Third party imports
#[allow(unused_imports)]
use ndarray::{Array, Array1, Ix1, Ix2};
// First party imports
use traits::function::*;

pub struct OrbitalExact<'a, T: 'a>
where T: ?Sized + Fn(&Array1<f64>) -> f64
{
    parms: Array1<f64>,
    basis_set: Vec<&'a T>
}

impl<'a, T> OrbitalExact<'a, T>
where T: ?Sized + Fn(&Array1<f64>) -> f64
{
   pub fn new(parms: Array1<f64>, basis_set: Vec<&'a T>) -> Self {
       Self{parms, basis_set}
   }
}

impl<'a, T> Function<f64> for OrbitalExact<'a, T>
where T: ?Sized + Fn(&Array1<f64>) -> f64 {

    type E = ();
    type D = Ix1;

    fn value(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Self::E> {
        let basis_vals = self.basis_set.iter().map(|x| x(cfg)).collect();
        Ok((Array1::from_vec(basis_vals) * &self.parms).scalar_sum())
    }
}
