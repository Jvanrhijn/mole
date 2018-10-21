// Defines various wave function representations, e.g. Jastrow-Slater
// third party imports
use ndarray::{Array1, Ix2, Array2};
// first party imports
use traits::differentiate::*;
use traits::function::Function;
use traits::wavefunction::WaveFunction;
use jastrow::JastrowFactor;
use determinant::Slater;
use orbitals::*;
use error::{Error};

/// Jastrow-Slater form wave function:
/// $\psi(x) = J(\alpha)\sum_{i=1}^{N_e}c_i D_i$.
/// The Jastrow-Slater wave function combines the anti-symmetrization properties of
/// Slater determinants with a prefactor that takes care of electron-electron cusp
/// conditions. The functional form is tailored to be easy to evaluate and differentiate
/// with respect to its parameters.
pub struct JastrowSlater {
    ci_coeffs: Array1<f64>,
    orb_coeffs: Array1<f64>,
    jastrow: JastrowFactor
}

impl JastrowSlater {
    pub fn new(cis: Array1<f64>, orbs: Array1<f64>, jas: JastrowFactor) -> Self {
        Self{ci_coeffs: cis, orb_coeffs: orbs, jastrow: jas}
    }
}

/// Single Slater determinant wave function:
/// $\psi(x) = \langle x | \hat{a}_{k_1}\ldots\hat{a}_{k_{N_e}} | 0 \rangle$.
/// This wave function is currently used for testing, but will be removed once the Jastrow-Slater
/// wave function is properly implemented.
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

impl<'a, T> Differentiate for SingleDeterminant<'a, T>
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

}

impl<'a, T> WaveFunction for SingleDeterminant<'a, T>
where T: ?Sized + Fn(&Array1<f64>) -> (f64, f64)
{
   fn num_electrons(&self) -> usize {
       self.det.num_electrons()
   }
}
