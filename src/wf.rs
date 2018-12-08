// Defines various wave function representations, e.g. Jastrow-Slater
// third party imports
use ndarray::{Array1, Ix2, Array2};
// first party imports
use traits::differentiate::*;
use traits::function::Function;
use traits::wavefunction::WaveFunction;
use traits::cache::Cache;
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

impl<'a, T> Cache<Array2<f64>> for SingleDeterminant<'a, T>
where T: ?Sized + Fn(&Array1<f64>) -> (f64, f64)
{
    type A = Array2<f64>;
    type V = (f64, f64);
    type U = usize;

    fn refresh(&mut self, new: &Self::A) {
        self.det.refresh(new);
    }

    fn update_inplace(&mut self, ud: Self::U, new: &Self::A) {
        self.det.update_inplace(ud, new);
    }

    fn update(&self, ud: Self::U, new: &Self::A) -> (Vec<Self::A>, Self::V) {
        self.det.update(ud, new)
    }

    fn set_cache(&mut self, storage: Vec<Self::A>, value: Self::V) {
        self.det.set_cache(storage, value);
    }

    fn current_value(&self) -> Self::V {
        self.det.current_value()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use math::basis;

    #[test]
    fn single_det_one_basis_function() {
        let basis = vec![
            Box::new(basis::hydrogen_1s)
        ];
        let orbital = Orbital::new(array![1.0], &basis);
        let mut wf = SingleDeterminant::new(vec![orbital]);
        let config = array![[1.0, 0.0, 0.0]];
        let config_slice = array![1.0, 0.0, 0.0];
        wf.refresh(&config);
        let cur_value = wf.current_value().0;
        assert_eq!(cur_value, basis::hydrogen_1s(&config_slice).0);
    }
}
