// Defines various wave function representations, e.g. Jastrow-Slater
// third party imports
use ndarray::{Array1, Ix2, Array2};
// first party imports
#[allow(unused_imports)]
use traits::wavefunction::*;
use traits::function::Function;
use jastrow::JastrowFactor;
use determinant::Slater;
use orbitals::*;
use error::{Error};

#[allow(dead_code)]
pub struct JastrowSlater {
    ci_coeffs: Array1<f64>,
    orb_coeffs: Array1<f64>,
    jastrow: JastrowFactor
}

#[allow(dead_code)]
impl JastrowSlater {
    pub fn new(cis: Array1<f64>, orbs: Array1<f64>, jas: JastrowFactor) -> Self {
        Self{ci_coeffs: cis, orb_coeffs: orbs, jastrow: jas}
    }
}

pub struct SingleDeterminant<'a, T>
where T: 'a + ?Sized + Fn(&Array1<f64>) -> (f64, f64)
{
    det: Slater<Orbital<'a, T>>,
}

impl<'a, T> SingleDeterminant<'a, T>
where T: ?Sized + Fn(&Array1<f64>) -> (f64, f64)
{
    pub fn new(orbs: Vec<Orbital<'a, T>>) -> Self {
        Self{det: Slater::new(orbs)}
    }
}

impl<'a, T> Function<f64> for SingleDeterminant<'a, T>
where T: ?Sized + Fn(&Array1<f64>) -> (f64, f64)
{
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        self.det.value(cfg)
    }
}

impl<'a, T> WaveFunction for SingleDeterminant<'a, T>
where T: ?Sized + Fn(&Array1<f64>) -> (f64, f64)
{
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Array2<f64> {
        let shape = cfg.shape();
        Array2::<f64>::ones((shape[0], shape[1]))
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        // TODO implement efficienctly
        self.det.laplacian(cfg)
    }

    fn num_electrons(&self) -> usize {
        self.det.num_electrons()
    }
}
