// Defines various wave function representations, e.g. Jastrow-Slater
use std::collections::VecDeque;
// third party imports
use ndarray::{Array2, Axis, Ix2};
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
#[derive(Clone)]
pub struct SingleDeterminant<T>
where
    T: BasisSet,
{
    det: Slater<Orbital<T>>,
}

impl<T> SingleDeterminant<T>
where
    T: BasisSet,
{
    pub fn new(orbs: Vec<Orbital<T>>) -> Self {
        Self {
            det: Slater::new(orbs),
        }
    }
}

impl<T> Function<f64> for SingleDeterminant<T>
where
    T: BasisSet,
{
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        self.det.value(cfg)
    }
}

impl<T> Differentiate for SingleDeterminant<T>
where
    T: BasisSet,
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
    T: BasisSet,
{
    fn num_electrons(&self) -> usize {
        self.det.num_electrons()
    }
}

impl<T> Cache<Array2<f64>> for SingleDeterminant<T>
where
    T: BasisSet,
{
    type A = Array2<f64>;
    type V = (f64, Array2<f64>, f64);
    type OV = (Option<f64>, Option<Array2<f64>>, Option<f64>);
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

    fn enqueued_value(&self) -> Self::OV {
        self.det.enqueued_value()
    }
}

// TODO: generalize to multi-determinant expansions
pub struct JastrowSlater<T: BasisSet> {
    det_up: SingleDeterminant<T>,
    det_down: SingleDeterminant<T>,
    jastrow: JastrowFactor,
    num_up: usize,
    num_down: usize,
    value_cache: VecDeque<f64>,
    grad_cache: VecDeque<Array2<f64>>,
    lapl_cache: VecDeque<f64>,
}

impl<T: BasisSet> JastrowSlater<T> {
    pub fn new(
        det_up: SingleDeterminant<T>,
        det_down: SingleDeterminant<T>,
        jastrow: JastrowFactor,
    ) -> Self {
        let num_up = det_up.num_electrons();
        let num_down = det_down.num_electrons();
        let value_cache = VecDeque::from(vec![1.0]);
        let grad_cache = VecDeque::from(vec![Array2::<f64>::ones((num_up + num_down, 3))]);
        let lapl_cache = VecDeque::from(vec![0.0]);
        Self {
            det_up,
            det_down,
            jastrow,
            num_up,
            num_down,
            value_cache,
            grad_cache,
            lapl_cache,
        }
    }

    fn split_config(&self, cfg: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let cfg_up = cfg.slice(s![..self.num_up, ..]).to_owned();
        let cfg_down = cfg.slice(s![self.num_up.., ..]).to_owned();
        (cfg_up, cfg_down)
    }
}

impl<T: BasisSet> Function<f64> for JastrowSlater<T> {
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        Ok(self.jastrow.value(cfg)?
            * self.det_up.value(&cfg_up)?
            * self.det_down.value(&cfg_down)?)
    }
}

impl<T: BasisSet> Differentiate for JastrowSlater<T> {
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Result<Array2<f64>, Error> {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        let det_up_val = self.det_up.value(&cfg_up)?;
        let det_down_val = self.det_down.value(&cfg_down)?;
        let jas_val = self.jastrow.value(cfg)?;

        let grad_det_up = self.det_up.gradient(&cfg_up)?;
        let grad_det_down = self.det_down.gradient(&cfg_down)?;
        let (grad_jas_up, grad_jas_down) = self.split_config(&self.jastrow.gradient(cfg)?);

        // apply prod rule:
        // $\nabla_\uparrow \psi = D^\downarrow(D^\uparrow\nabla_\uparrow J + J\nabla_\uparrow D^\uparrow$
        // and similar for $\nabla_\downarrow \psi$
        // then $\nabla \psi = (\nabla_\uparrow \psi, \nabla_\downarrow \psi)$.
        let grad_up = det_down_val * (det_up_val * &grad_jas_up + jas_val * &grad_det_up);
        let grad_down = det_up_val * (det_down_val * &grad_jas_down + jas_val * &grad_det_down);

        // concatenate gradients
        Ok(stack![Axis(0), grad_up, grad_down])
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        let det_up_val = self.det_up.value(&cfg_up)?;
        let det_down_val = self.det_down.value(&cfg_down)?;
        let jas_val = self.jastrow.value(cfg)?;

        let lapl_det_up = self.det_up.laplacian(&cfg_up)?;
        let lapl_det_down = self.det_down.laplacian(&cfg_down)?;
        let lapl_jas = self.jastrow.laplacian(&cfg_down)?;

        let grad_det_up = self.det_up.gradient(&cfg_up)?;
        let grad_det_down = self.det_down.gradient(&cfg_down)?;
        let grad_jas = self.jastrow.gradient(cfg)?;

        let (grad_jas_up, grad_jas_down) = self.split_config(&grad_jas);
        // Laplacian formula for $\psi = J(r)\D^\uparrow D^\downarrow$:
        // $\Delta = \Delta_up + \Delta_down$, and
        // $\Delta (uv) = u\Delta v + v\Delta u + \nabla u \dot \nabla v$, so
        // $\Delta_\uparrow \psi = D^\downarrow(D^\uparrow\Delta_\uparrow J
        // + \nabla\uparrow J \dot \nabla_\uparrow D^\uparrow$
        // similar for $\Delta_\downarrow$
        let laplacian = det_down_val * det_up_val * lapl_jas
            + det_down_val * (&grad_jas_up * &grad_det_up).scalar_sum()
            + det_up_val * (&grad_jas_down * &grad_det_down).scalar_sum();
        Ok(laplacian)
    }
}

impl<T: BasisSet> WaveFunction for JastrowSlater<T> {
    fn num_electrons(&self) -> usize {
        self.num_up + self.num_down
    }
}

impl<T: BasisSet> Cache<Array2<f64>> for JastrowSlater<T> {
    type A = Array2<f64>;
    type V = (f64, Array2<f64>, f64);
    type OV = (Option<f64>, Option<Array2<f64>>, Option<f64>);
    type U = usize;

    fn refresh(&mut self, cfg: &Self::A) {
        self.det_up.refresh(cfg);
        self.det_down.refresh(cfg);
        self.jastrow.refresh(cfg);
    }

    fn enqueue_update(&mut self, ud: Self::U, cfg: &Array2<f64>) {
        if ud < self.num_up {
            self.det_up.enqueue_update(ud, cfg);
        } else {
            self.det_down.enqueue_update(ud - self.num_up, cfg);
        }
        self.jastrow.enqueue_update(ud, cfg);
        let (det_up_v, det_up_g, det_up_l) = match self.det_up.enqueued_value() {
            (Some(v), Some(g), Some(l)) => (v, g, l),
            _ => self.det_up.current_value(),
        };
        let (det_down_v, det_down_g, det_down_l) = match self.det_down.enqueued_value() {
            (Some(v), Some(g), Some(l)) => (v, g, l),
            _ => self.det_down.current_value(),
        };
        let (jas_v, jas_g, jas_l) = match self.jastrow.enqueued_value() {
            (Some(v), Some(g), Some(l)) => (v, g, l),
            _ => self.jastrow.current_value(),
        };
        // Cache wave function value
        self.value_cache.push_back(det_up_v * det_down_v * jas_v);
        // gradient w.r.t. spin-up electron coordinates
        let (jas_g_up, jas_g_down) = self.split_config(&jas_g);
        // gradient w.r.t. spin-up electron coordinates
        let grad_up = det_up_v * det_down_v * &jas_g_up + det_down_v * jas_v * &det_up_g;
        // gradient w.r.t. spin-down electron coordinates
        let grad_down = det_down_v * det_up_v * &jas_g_down + det_up_v * jas_v * &det_down_g;
        // full gradient
        let grad = stack![Axis(0), grad_up, grad_down];
        self.grad_cache.push_back(grad);
        // laplacian computation
        let laplacian = det_up_v * det_down_v * jas_l
            + det_down_v * (&jas_g_up * &det_up_g).scalar_sum()
            + det_up_v * (&jas_g_down * &det_down_g).scalar_sum();
        self.lapl_cache.push_back(laplacian);
    }

    fn push_update(&mut self) {
        self.det_up.push_update();
        self.det_down.push_update();
        self.jastrow.push_update();
        self.value_cache.pop_front();
        self.grad_cache.pop_front();
        self.lapl_cache.pop_front();
    }

    fn flush_update(&mut self) {
        self.det_up.flush_update();
        self.det_down.flush_update();
        self.jastrow.flush_update();
        self.value_cache.pop_back();
        self.grad_cache.pop_back();
        self.lapl_cache.pop_back();
    }

    fn current_value(&self) -> Self::V {
        match (
            self.value_cache.front(),
            self.grad_cache.front(),
            self.lapl_cache.front(),
        ) {
            (Some(&v), Some(g), Some(&l)) => (v, g.clone(), l),
            _ => panic!("No value stored in JastrowSlater cache"),
        }
    }

    fn enqueued_value(&self) -> Self::OV {
        (
            self.value_cache
                .back()
                .and(Some(*self.value_cache.back().unwrap())),
            self.grad_cache
                .back()
                .and(Some(self.grad_cache.back().unwrap().clone())),
            self.lapl_cache
                .back()
                .and(Some(*self.lapl_cache.back().unwrap())),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use basis::{gaussian, GaussianBasis, Hydrogen1sBasis};
    const EPS: f64 = 1e-15;

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

    #[test]
    fn jastrow_slater_single_det() {
        use ndarray::{arr2, Zip};
        use ndarray_linalg::Norm;
        let basis = GaussianBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0]);
        let orbital = Orbital::new(array![[1.0]], basis);

        let det_up = SingleDeterminant::new(vec![orbital]);
        let det_down = det_up.clone();

        let jastrow = JastrowFactor::new(array![0.5, 0.5, 0.0], 2);

        let wf = JastrowSlater::new(det_up, det_down, jastrow);

        let cfg = array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let x1 = cfg.slice(s![0, ..]).to_owned();
        let x2 = cfg.slice(s![1, ..]).to_owned();
        let x12 = &x1 - &x2;
        let r12 = x12.norm_l2();

        let (value_orb_1, grad_orb_1, lapl_orb_1) = gaussian(&x1, 1.0);
        let (value_orb_2, grad_orb_2, lapl_orb_2) = gaussian(&x2, 1.0);

        let value_jastr = (0.5 * r12 / (1.0 + 0.5 * r12)).exp();
        let grad_f_1 = (-2.0 / (2.0 + r12).powi(2) / r12 * &x12)
            .into_shape((1, 3))
            .unwrap();
        let grad_f_2 = (2.0 / (2.0 + r12).powi(2) / r12 * &x12)
            .into_shape((1, 3))
            .unwrap();
        let grad_f = stack![Axis(0), grad_f_1, grad_f_2];
        let grad_jastr = value_jastr * grad_f;

        let test_value = value_jastr * value_orb_1 * value_orb_2;
        let value = wf.value(&cfg).unwrap();
        assert_eq!(test_value, value);

        let test_grad_1 = (value_orb_2 * value_jastr * (value_orb_1 * &grad_f_1 + &grad_orb_1))
            .into_shape((1, 3))
            .unwrap();
        let test_grad_2 = (value_orb_1 * value_jastr * (value_orb_2 * &grad_f_2 + &grad_orb_2))
            .into_shape((1, 3))
            .unwrap();

        let test_grad = stack![Axis(0), test_grad_1, test_grad_2];
        let grad = wf.gradient(&cfg).unwrap();
        assert!(test_grad.all_close(&grad, EPS));

        //TODO: test laplacian
    }
}
