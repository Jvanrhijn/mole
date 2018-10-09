// Defines various wave function representations, e.g. Jastrow-Slater
// third party imports
use ndarray::{Array1, Ix2, Ix1, Array2};
// first party imports
#[allow(unused_imports)]
use traits::wavefunction::*;
use traits::function::Function;
use jastrow::JastrowFactor;
use determinant::Determinant;
use orbitals::*;

#[derive(Debug)]
pub struct JastrowSlater {
    ci_coeffs: Array1<f64>,
    orb_coeffs: Array1<f64>,
    jastrow: JastrowFactor
}

impl JastrowSlater {
    pub fn new(cis: Array1<f64>, orbs: Array1<f64>, jas: JastrowFactor) -> Self {
        JastrowSlater{ci_coeffs: cis, orb_coeffs: orbs, jastrow: jas}
    }
}

struct SingleDeterminant<T: Function<f64, D=Ix1>> {
    det: Determinant<OrbitalExact<T>>,
}

impl<T: Function<f64, D=Ix1>> SingleDeterminant<T> {
    pub fn new(orbs: Vec<OrbitalExact<T>>) -> Self {
        Self{det: Determinant::new(orbs)}
    }
}

impl<T: Function<f64, D=Ix1>> Function<f64> for SingleDeterminant<T> {

    type E = ();
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Self::E> {
        Ok(1.0)
    }
}
