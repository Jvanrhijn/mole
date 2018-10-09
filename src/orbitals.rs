// Standard imports
use std::vec::Vec;
// Third party imports
#[allow(unused_imports)]
use ndarray::{Array, Array1, Ix1};
// First party imports
use traits::function::*;

pub struct OrbitalExact<T: Function<f64>> {
    parms: Array1<f64>,
    basis_set: Vec<T>
}

impl<T: Function<f64>> OrbitalExact<T> {
   pub fn new(parms: Array1<f64>, basis_set: Vec<T>) -> Self {
       Self{parms, basis_set}
   }
}

impl<T> Function<f64> for OrbitalExact<T> where T: Function<f64, D=Ix1> {

    type E = ();
    type D = Ix1;

    fn value(&self, cfg: &Array<f64, Ix1>) -> Result<f64, Self::E> {
        let basis_vals = self.basis_set.iter().map(|x| x.value(cfg).unwrap()).collect();
        Ok((Array1::from_vec(basis_vals) * &self.parms).scalar_sum())
    }
}


