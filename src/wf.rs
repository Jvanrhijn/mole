// Defines various wave function representations, e.g. Jastrow-Slater
// third party imports
use ndarray::{Array1, Ix2, Ix1, Array2};
use ndarray_linalg::error::LinalgError;
// first party imports
#[allow(unused_imports)]
use traits::wavefunction::*;
use traits::function::Function;
use jastrow::JastrowFactor;
use determinant::Determinant;
use orbitals::*;

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

pub struct SingleDeterminant<'a, T>
where T: 'a + ?Sized + Fn(&Array1<f64>) -> f64
{
    det: Determinant<OrbitalExact<'a, T>>,
}

impl<'a, T> SingleDeterminant<'a, T>
where T: ?Sized + Fn(&Array1<f64>) -> f64
{
    pub fn new(orbs: Vec<OrbitalExact<'a, T>>) -> Self {
        Self{det: Determinant::new(orbs)}
    }
}

impl<'a, T> Function<f64> for SingleDeterminant<'a, T>
where T: ?Sized + Fn(&Array1<f64>) -> f64
{

    type E = LinalgError;
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Self::E> {
        self.det.value(cfg)
    }
}
