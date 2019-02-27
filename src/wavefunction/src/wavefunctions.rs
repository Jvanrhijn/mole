// Defines various wave function representations, e.g. Jastrow-Slater
use std::collections::VecDeque;
// third party imports
use ndarray::{Array1, Array2, Axis, Ix2};
// first party imports
use crate::determinant::Slater;
use crate::error::Error;
use crate::jastrow::JastrowFactor;
use crate::orbitals::Orbital;
use crate::traits::{Cache, Differentiate, Function, WaveFunction};
use basis::BasisSet;

type Vgl = (f64, Array2<f64>, f64);
type Ovgl = (Option<f64>, Option<Array2<f64>>, Option<f64>);

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

impl<T> Cache for SingleDeterminant<T>
where
    T: BasisSet,
{
    type U = usize;

    fn refresh(&mut self, new: &Array2<f64>) {
        self.det.refresh(new);
    }

    fn enqueue_update(&mut self, ud: Self::U, new: &Array2<f64>) {
        self.det.enqueue_update(ud, new);
    }

    fn push_update(&mut self) {
        self.det.push_update();
    }

    fn flush_update(&mut self) {
        self.det.flush_update();
    }

    fn current_value(&self) -> Vgl {
        self.det.current_value()
    }

    fn enqueued_value(&self) -> Ovgl {
        self.det.enqueued_value()
    }
}

// TODO: generalize to multi-determinant expansions
/// Jastrow-Slater form wave function:
/// $\psi(x) = J(\alpha)D^\uparrow D^\downarrow$.
/// The Jastrow-Slater wave function combines the anti-symmetrization properties of
/// Slater determinants with a prefactor that takes care of electron-electron cusp
/// conditions. The functional form is tailored to be easy to evaluate and differentiate
/// with respect to its parameters.
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
        parms: Array1<f64>,
        mut orbitals: Vec<Orbital<T>>,
        scal: f64,
        num_up: usize,
    ) -> Self {
        let num_elec = orbitals.len();
        let down_orbs: Vec<_> = orbitals.drain(num_up..).collect();
        let up_orbs = orbitals;
        let det_up = SingleDeterminant::new(up_orbs);
        let det_down = SingleDeterminant::new(down_orbs);
        let jastrow = JastrowFactor::new(parms, num_elec, scal, num_up);
        let value_cache = VecDeque::from(vec![1.0]);
        let grad_cache = VecDeque::from(vec![Array2::<f64>::ones((num_elec, 3))]);
        let lapl_cache = VecDeque::from(vec![0.0]);
        Self {
            det_up,
            det_down,
            jastrow,
            num_up,
            num_down: num_elec - num_up,
            value_cache,
            grad_cache,
            lapl_cache,
        }
    }

    pub fn from_components(
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
        let lapl_jas = self.jastrow.laplacian(cfg)?;

        let grad_det_up = self.det_up.gradient(&cfg_up)?;
        let grad_det_down = self.det_down.gradient(&cfg_down)?;
        let grad_jas = self.jastrow.gradient(cfg)?;

        let (grad_jas_up, grad_jas_down) = self.split_config(&grad_jas);

        // Laplacian formula for $\psi = J(r)\D^\uparrow D^\downarrow$:
        // see theory/jastrowslater.tex for derivation of formula
        let laplacian = 2.0 * det_down_val * (&grad_det_up * &grad_jas_up).scalar_sum()
            + 2.0 * det_up_val * (&grad_det_down * &grad_jas_down).scalar_sum()
            + det_up_val * det_down_val * lapl_jas
            + jas_val * (det_up_val * lapl_det_down + det_down_val * lapl_det_up);

        Ok(laplacian)
    }
}

impl<T: BasisSet> WaveFunction for JastrowSlater<T> {
    fn num_electrons(&self) -> usize {
        self.num_up + self.num_down
    }
}

impl<T: BasisSet> Cache for JastrowSlater<T> {
    type U = usize;

    fn refresh(&mut self, cfg: &Array2<f64>) {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        self.det_up.refresh(&cfg_up);
        self.det_down.refresh(&cfg_down);
        self.jastrow.refresh(cfg);
        // TODO get rid of calls to self.value/gradient/laplacian
        *self.value_cache.front_mut().expect("Value cache empty")
            = self.value(cfg).expect("Failed to take Jastrow Slater value");
        *self.grad_cache.front_mut().expect("Gradient cache empty")
            = self.gradient(cfg).expect("Failed to take Jastrow Slater gradient");
        *self.lapl_cache.front_mut().expect("Laplacian cache empty")
            = self.laplacian(cfg).expect("Failed to take Jastrow Slater laplacian");
    }

    fn enqueue_update(&mut self, ud: Self::U, cfg: &Array2<f64>) {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        if ud < self.num_up {
            self.det_up.enqueue_update(ud, &cfg_up);
        } else {
            self.det_down.enqueue_update(ud - self.num_up, &cfg_down);
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
        let laplacian = 2.0 * det_down_v * (&det_up_g * &jas_g_up).scalar_sum()
            + 2.0 * det_up_v * (&det_down_g * &jas_g_down).scalar_sum()
            + det_up_v * det_down_v * jas_l
            + jas_v * (det_up_v * det_down_l + det_down_v * det_up_l);
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

    fn current_value(&self) -> Vgl {
        match (
            self.value_cache.front(),
            self.grad_cache.front(),
            self.lapl_cache.front(),
        ) {
            (Some(&v), Some(g), Some(&l)) => (v, g.clone(), l),
            _ => panic!("No value stored in JastrowSlater cache"),
        }
    }

    fn enqueued_value(&self) -> Ovgl {
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
    use basis::{GaussianBasis, Hydrogen1sBasis};

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
    fn jastrow_slater_grad_laplacian() {
        use crate::util::grad_laplacian_finite_difference;
        let basis = GaussianBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0, 2.0, 3.0]);
        let orbitals = vec![
            Orbital::new(array![[1.0, 0.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 1.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 0.0, 1.0]], basis.clone()),
        ];
        let wave_function = JastrowSlater::new(array![1.0], orbitals, 0.1, 2);

        let cfg = array![[1.0, 2.0, 3.0], [0.1, -0.5, 0.2], [-1.2, -0.8, 0.3]];
        let (grad_fd, laplac_fd) =
            grad_laplacian_finite_difference(&wave_function, &cfg, 1e-4).unwrap();
        assert!(grad_fd.all_close(&wave_function.gradient(&cfg).unwrap(), 1e-5));
        assert!((laplac_fd - wave_function.laplacian(&cfg).unwrap()).abs() < 1e-5);
    }

    #[test]
    fn jastrow_slater_cache() {
        let basis = GaussianBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0, 2.0, 3.0]);
        let orbitals = vec![
            Orbital::new(array![[1.0, 0.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 1.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 0.0, 1.0]], basis.clone()),
        ];
        let mut wave_function = JastrowSlater::new(array![1.0], orbitals, 0.1, 2);

        let mut cfg = array![[1.0, 2.0, 3.0], [0.1, -0.5, 0.2], [-1.2, -0.8, 0.3]];

        let value = wave_function.value(&cfg).unwrap();
        let grad = wave_function.gradient(&cfg).unwrap();
        let laplacian = wave_function.laplacian(&cfg).unwrap();
        wave_function.refresh(&cfg);

        assert_eq!(wave_function.current_value().0, value);
        assert_eq!(wave_function.current_value().1, grad);
        assert_eq!(wave_function.current_value().2, laplacian);

        // change cfg
        cfg[[0, 0]] = 2.0;
        
        let value = wave_function.value(&cfg).unwrap();
        let grad = wave_function.gradient(&cfg).unwrap();
        let laplacian = wave_function.laplacian(&cfg).unwrap();
        wave_function.enqueue_update(0, &cfg);
        wave_function.push_update();

        assert_eq!(wave_function.current_value().0, value);
        assert!(wave_function.current_value().1.all_close(&grad, 1e-14));
        assert_eq!(wave_function.current_value().2, laplacian);
    }
}
