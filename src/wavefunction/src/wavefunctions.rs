// Defines various wave function representations, e.g. Jastrow-Slater
// third party imports
use ndarray::{Array2, Ix2};
// first party imports
use crate::determinant::Slater;
use crate::error::Error;
use crate::jastrow::JastrowFactor;
use crate::orbitals::Orbital;
use crate::traits::{Cache, Differentiate, Function, WaveFunction};
use basis::BasisSet;

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
pub struct SingleDeterminant<T>
where
    T: BasisSet
{
    det: Slater<Orbital<T>>,
}

impl<T> SingleDeterminant<T>
where
    T: BasisSet
{
    pub fn new(orbs: Vec<Orbital<T>>) -> Self {
        Self {
            det: Slater::new(orbs),
        }
    }
}

impl<T> Function<f64> for SingleDeterminant<T>
where
    T: BasisSet
{
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        self.det.value(cfg)
    }
}

impl<T> Differentiate for SingleDeterminant<T>
where
    T: BasisSet
{
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Result<Array2<f64>, Error> {
        self.det.gradient(cfg)
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        self.det.laplacian(cfg)
    }
}

impl<T> WaveFunction for SingleDeterminant<T>
where
    T: BasisSet
{
    fn num_electrons(&self) -> usize {
        self.det.num_electrons()
    }
}

impl<T> Cache<Array2<f64>> for SingleDeterminant<T>
where
    T: BasisSet
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

pub struct JastrowSlater {
    //det_up: SingleDeterminant,
    //det_down: SingleDeterminant,
    jastrow: JastrowFactor,
}

#[cfg(test)]
mod tests {
    use super::*;
    use basis::Hydrogen1sBasis;

    #[test]
    fn single_det_one_basis_function() {
        let basis = Hydrogen1sBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0]);

        let orbital = Orbital::new(array![[1.0]], basis);
        let mut wf = SingleDeterminant::new(vec![orbital]);
        let config = array![[1.0, 0.0, 0.0]];
        let config_slice = array![1.0, 0.0, 0.0];

        wf.refresh(&config);

        let cur_value = wf.current_value().0;
        let cur_grad = wf.current_value().1;
        let cur_laplac = wf.current_value().2;

        assert_eq!(cur_value, basis::hydrogen_1s(&config_slice, 1.0).0);
        assert_eq!(
            cur_grad.slice(s![0, ..]),
            basis::hydrogen_1s(&config_slice, 1.0).1
        );
        assert_eq!(cur_laplac, basis::hydrogen_1s(&config_slice, 1.0).2);
    }
}
