// Defines various wave function representations, e.g. Jastrow-Slater
// third party imports
use ndarray::{Array1, Ix2, Array2};
// first party imports
use crate::orbitals::Orbital;
use crate::traits::{Function, WaveFunction, Cache, Differentiate};
use crate::determinant::Slater;
use crate::error::Error;
use basis::Vgl;

/// Jastrow-Slater form wave function:
/// $\psi(x) = J(\alpha)\sum_{i=1}^{N_e}c_i D_i$.
/// The Jastrow-Slater wave function combines the anti-symmetrization properties of
/// Slater determinants with a prefactor that takes care of electron-electron cusp
/// conditions. The functional form is tailored to be easy to evaluate and differentiate
/// with respect to its parameters.
//#[allow(dead_code)]
//pub struct JastrowSlater {
//    ci_coeffs: Array1<f64>,
//    orb_coeffs: Array1<f64>,
//    jastrow: JastrowFactor
//}
//
//impl JastrowSlater {
//    pub fn new(cis: Array1<f64>, orbs: Array1<f64>, jas: JastrowFactor) -> Self {
//        unimplemented!()
//        //Self{ci_coeffs: cis, orb_coeffs: orbs, jastrow: jas}
//    }
//}

/// Single Slater determinant wave function:
/// $\psi(x) = \langle x | \hat{a}_{k_1}\ldots\hat{a}_{k_{N_e}} | 0 \rangle$.
/// This wave function is currently used for testing, but will be removed once the Jastrow-Slater
/// wave function is properly implemented.
pub struct SingleDeterminant<'a, T>
    where T: 'a + ?Sized + Fn(&Array1<f64>) -> Vgl
{
    det: Slater<Orbital<'a, T>>,
}

impl<'a, T> SingleDeterminant<'a, T>
    where T: ?Sized + Fn(&Array1<f64>) -> Vgl
{
    pub fn new(orbs: Vec<Orbital<'a, T>>) -> Self {
        Self{det: Slater::new(orbs)}
    }
}

impl<'a, T> Function<f64> for SingleDeterminant<'a, T>
    where T: ?Sized + Fn(&Array1<f64>) -> Vgl
{
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        self.det.value(cfg)
    }
}

impl<'a, T> Differentiate for SingleDeterminant<'a, T>
    where T: ?Sized + Fn(&Array1<f64>) -> Vgl
{
    type D = Ix2;

    fn gradient(&self, _cfg: &Array2<f64>) -> Result<Array2<f64>, Error> {
        unimplemented!()
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        // TODO implement efficienctly
        self.det.laplacian(cfg)
    }

}

impl<'a, T> WaveFunction for SingleDeterminant<'a, T>
    where T: ?Sized + Fn(&Array1<f64>) -> Vgl
{
    fn num_electrons(&self) -> usize {
        self.det.num_electrons()
    }
}

impl<'a, T> Cache<Array2<f64>> for SingleDeterminant<'a, T>
    where T: ?Sized + Fn(&Array1<f64>) -> Vgl
{
    type A = Array2<f64>;
    type V = (f64, Array2<f64>, f64);
    type U = usize;

    fn refresh(&mut self, new: &Self::A) {
        self.det.refresh(new);
    }

    fn enqueue_update(&mut self, ud: Self::U, new: &Self::A) {
        self.det.enqueue_update(ud, new);
    }

    fn push_update(&mut self) {
        self.det.push_update();
    }

    fn flush_update(&mut self) {
        self.det.flush_update();
    }

    fn current_value(&self) -> Self::V {
        self.det.current_value()
    }

    fn enqueued_value(&self) -> Option<Self::V> {
        self.det.enqueued_value()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use basis;

    #[test]
    fn single_det_one_basis_function() {
        let basis: Vec<Box<basis::Func>> = vec![
            Box::new(|x| basis::hydrogen_1s(x, 1.0))
        ];
        let orbital = Orbital::new(array![1.0], &basis);
        let mut wf = SingleDeterminant::new(vec![orbital]);
        let config = array![[1.0, 0.0, 0.0]];
        let config_slice = array![1.0, 0.0, 0.0];
        wf.refresh(&config);
        let cur_value = wf.current_value().0;
        assert_eq!(cur_value, basis::hydrogen_1s(&config_slice, 1.0).0);
    }
}
