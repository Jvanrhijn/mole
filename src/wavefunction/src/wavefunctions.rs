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
use optimize::Optimize;

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

#[derive(Clone)]
pub struct SpinDeterminantProduct<T>
where
    T: BasisSet,
{
    det_up: Slater<Orbital<T>>,
    det_down: Slater<Orbital<T>>,
    num_up: usize,
    value_cache: VecDeque<f64>,
    grad_cache: VecDeque<Array2<f64>>,
    laplacian_cache: VecDeque<f64>,
}

impl<T: BasisSet> SpinDeterminantProduct<T> {
    pub fn new(mut orbitals: Vec<Orbital<T>>, num_up: usize) -> Self {
        let nelec = orbitals.len();
        let orb_up = orbitals.drain(0..num_up).collect();
        Self {
            det_up: Slater::new(orb_up),
            det_down: Slater::new(orbitals),
            num_up,
            value_cache: VecDeque::from(vec![1.0]),
            grad_cache: VecDeque::from(vec![Array2::zeros((nelec, 3))]),
            laplacian_cache: VecDeque::from(vec![1.0]),
        }
    }

    fn split_config(&self, cfg: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let cfg_up = cfg.slice(s![..self.num_up, ..]).to_owned();
        let cfg_down = cfg.slice(s![self.num_up.., ..]).to_owned();
        (cfg_up, cfg_down)
    }
}

impl<T: BasisSet> WaveFunction for SpinDeterminantProduct<T> {
    fn num_electrons(&self) -> usize {
        self.det_up.num_electrons() + self.det_down.num_electrons()
    }
}

impl<T: BasisSet> Function<f64> for SpinDeterminantProduct<T> {
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        Ok(self.det_up.value(&cfg_up)? * self.det_down.value(&cfg_down)?)
    }
}

impl<T: BasisSet> Differentiate for SpinDeterminantProduct<T> {
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Result<Array2<f64>, Error> {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        let det_up_v = self.det_up.value(&cfg_up)?;
        let det_down_v = self.det_down.value(&cfg_down)?;
        Ok(stack![
            Axis(0),
            det_down_v * self.det_up.gradient(&cfg_up)?,
            det_up_v * self.det_down.gradient(&cfg_down)?
        ])
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        let det_up_v = self.det_up.value(&cfg_up)?;
        let det_down_v = self.det_down.value(&cfg_down)?;
        Ok(det_down_v * self.det_up.laplacian(&cfg_up)?
            + det_up_v * self.det_down.laplacian(&cfg_down)?)
    }
}

impl<T: BasisSet> Cache for SpinDeterminantProduct<T> {
    type U = usize;

    fn refresh(&mut self, cfg: &Array2<f64>) {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        self.det_up.refresh(&cfg_up);
        self.det_down.refresh(&cfg_down);
        let (det_up_v, det_up_g, det_up_l) = self.det_up.current_value();
        let (det_down_v, det_down_g, det_down_l) = self.det_down.current_value();
        *self.value_cache.front_mut().expect("Value cache empty") = det_up_v * det_down_v;
        *self.grad_cache.front_mut().expect("Gradient cache empty") =
            stack![Axis(0), det_down_v * &det_up_g, det_up_v * &det_down_g];
        *self
            .laplacian_cache
            .front_mut()
            .expect("Laplacian cache empty") = det_up_v * det_down_l + det_down_v * det_up_l;
        self.flush_update();
    }

    fn enqueue_update(&mut self, ud: Self::U, cfg: &Array2<f64>) {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        if ud < self.num_up {
            self.det_up.enqueue_update(ud, &cfg_up);
        } else {
            self.det_down.enqueue_update(ud - self.num_up, &cfg_down);
        }
        let (det_up_v, det_up_g, det_up_l) = match self.det_up.enqueued_value() {
            (Some(v), Some(g), Some(l)) => (v, g, l),
            _ => self.det_up.current_value(),
        };
        let (det_down_v, det_down_g, det_down_l) = match self.det_down.enqueued_value() {
            (Some(v), Some(g), Some(l)) => (v, g, l),
            _ => self.det_down.current_value(),
        };
        self.value_cache.push_back(det_up_v * det_down_v);
        self.grad_cache.push_back(stack![
            Axis(0),
            det_down_v * &det_up_g,
            det_up_v * &det_down_g
        ]);
        self.laplacian_cache
            .push_back(det_up_v * det_down_l + det_down_v * det_up_l);
    }

    fn push_update(&mut self) {
        self.det_up.push_update();
        self.det_down.push_update();
        self.value_cache.pop_front();
        self.grad_cache.pop_front();
        self.laplacian_cache.pop_front();
    }

    fn flush_update(&mut self) {
        self.det_up.flush_update();
        self.det_down.flush_update();
        if self.value_cache.len() == 2 {
            self.value_cache.pop_back();
        }
        if self.grad_cache.len() == 2 {
            self.grad_cache.pop_back();
        }
        if self.laplacian_cache.len() == 2 {
            self.laplacian_cache.pop_back();
        }
    }

    fn current_value(&self) -> Vgl {
        match (
            self.value_cache.front(),
            self.grad_cache.front(),
            self.laplacian_cache.front(),
        ) {
            (Some(&v), Some(g), Some(&l)) => (v, g.clone(), l),
            _ => panic!("Attempt to retrieve value from empty cache"),
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
            self.laplacian_cache
                .back()
                .and(Some(*self.laplacian_cache.back().unwrap())),
        )
    }
}

// TODO: generalize to multi-determinant expansions
/// Jastrow-Slater form wave function:
/// $\psi(x) = J(\alpha)D^\uparrow D^\downarrow$.
/// The Jastrow-Slater wave function combines the anti-symmetrization properties of
/// Slater determinants with a prefactor that takes care of electron-electron cusp
/// conditions. The functional form is tailored to be easy to evaluate and differentiate
/// with respect to its parameters.
#[derive(Clone)]
pub struct JastrowSlater<T: BasisSet> {
    det: SpinDeterminantProduct<T>,
    jastrow: JastrowFactor,
    value_cache: VecDeque<f64>,
    grad_cache: VecDeque<Array2<f64>>,
    lapl_cache: VecDeque<f64>,
}

impl<T: BasisSet> JastrowSlater<T> {
    pub fn new(parms: Array1<f64>, orbitals: Vec<Orbital<T>>, scal: f64, num_up: usize) -> Self {
        let num_elec = orbitals.len();
        let jastrow = JastrowFactor::new(parms, num_elec, scal, num_up);
        let value_cache = VecDeque::from(vec![1.0]);
        let grad_cache = VecDeque::from(vec![Array2::<f64>::ones((num_elec, 3))]);
        let lapl_cache = VecDeque::from(vec![0.0]);
        Self {
            det: SpinDeterminantProduct::new(orbitals, num_up),
            jastrow,
            value_cache,
            grad_cache,
            lapl_cache,
        }
    }

    pub fn from_components(det: SpinDeterminantProduct<T>, jastrow: JastrowFactor) -> Self {
        let value_cache = VecDeque::from(vec![1.0]);
        let grad_cache = VecDeque::from(vec![Array2::<f64>::ones((det.num_electrons(), 3))]);
        let lapl_cache = VecDeque::from(vec![0.0]);
        Self {
            det,
            jastrow,
            value_cache,
            grad_cache,
            lapl_cache,
        }
    }
}

impl<T: BasisSet> Function<f64> for JastrowSlater<T> {
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        Ok(self.jastrow.value(cfg)? * self.det.value(cfg)?)
    }
}

impl<T: BasisSet> Differentiate for JastrowSlater<T> {
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Result<Array2<f64>, Error> {
        Ok(self.jastrow.gradient(cfg)? * self.det.value(cfg)?
            + self.jastrow.value(cfg)? * self.det.gradient(cfg)?)
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        let det_v = self.det.value(cfg)?;
        let det_g = self.det.gradient(cfg)?;
        let det_l = self.det.laplacian(cfg)?;
        let jas_v = self.jastrow.value(cfg)?;
        let jas_g = self.jastrow.gradient(cfg)?;
        let jas_l = self.jastrow.laplacian(cfg)?;
        Ok(det_v * jas_l + jas_v * det_l + 2.0 * (&det_g * &jas_g).scalar_sum())
    }
}

impl<T: BasisSet> WaveFunction for JastrowSlater<T> {
    fn num_electrons(&self) -> usize {
        self.det.num_electrons()
    }
}

impl<T: BasisSet> Cache for JastrowSlater<T> {
    type U = usize;

    fn refresh(&mut self, cfg: &Array2<f64>) {
        self.det.refresh(cfg);
        self.jastrow.refresh(cfg);
        // TODO get rid of calls to self.value/gradient/laplacian
        *self.value_cache.front_mut().expect("Value cache empty") = self
            .value(cfg)
            .expect("Failed to take Jastrow Slater value");
        *self.grad_cache.front_mut().expect("Gradient cache empty") = self
            .gradient(cfg)
            .expect("Failed to take Jastrow Slater gradient");
        *self.lapl_cache.front_mut().expect("Laplacian cache empty") = self
            .laplacian(cfg)
            .expect("Failed to take Jastrow Slater laplacian");
        self.flush_update();
    }

    fn enqueue_update(&mut self, ud: Self::U, cfg: &Array2<f64>) {
        self.det.enqueue_update(ud, cfg);
        self.jastrow.enqueue_update(ud, cfg);
        let (det_v, det_g, det_l) = match self.det.enqueued_value() {
            (Some(v), Some(g), Some(l)) => (v, g, l),
            _ => unreachable!(),
        };
        let (jas_v, jas_g, jas_l) = match self.jastrow.enqueued_value() {
            (Some(v), Some(g), Some(l)) => (v, g, l),
            _ => unreachable!(),
        };
        self.value_cache.push_back(det_v * jas_v);
        self.grad_cache.push_back(det_v * &jas_g + jas_v * &det_g);
        let laplacian = det_v * jas_l + jas_v * det_l + 2.0 * (&det_g * &jas_g).scalar_sum();
        self.lapl_cache.push_back(laplacian);
    }

    fn push_update(&mut self) {
        self.det.push_update();
        self.jastrow.push_update();
        self.value_cache.pop_front();
        self.grad_cache.pop_front();
        self.lapl_cache.pop_front();
    }

    fn flush_update(&mut self) {
        self.det.flush_update();
        self.jastrow.flush_update();
        if self.value_cache.len() == 2 {
            self.value_cache.pop_back();
        }
        if self.grad_cache.len() == 2 {
            self.grad_cache.pop_back();
        }
        if self.lapl_cache.len() == 2 {
            self.lapl_cache.pop_back();
        }
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

impl<T: BasisSet> Optimize for JastrowSlater<T> {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Array1<f64> {
        self.jastrow.parameter_gradient(cfg)
    }

    fn update_parameters(&mut self, deltap: &Array1<f64>) {
        self.jastrow.update_parameters(deltap);
    }

    fn parameters(&self) -> &Array1<f64> {
        self.jastrow.parameters()
    }

    fn num_parameters(&self) -> usize {
        self.jastrow.num_parameters()
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

        assert!((cur_value - basis::hydrogen_1s(&config_slice, 1.0).0).abs() < 1e-15);
        assert_eq!(
            cur_grad.slice(s![0, ..]),
            basis::hydrogen_1s(&config_slice, 1.0).1
        );
        assert!((cur_laplac - basis::hydrogen_1s(&config_slice, 1.0).2).abs() < 1e-15);
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
        let wave_function = JastrowSlater::new(array![1.0, 0.01, 0.01], orbitals, 0.1, 2);

        let cfg = array![[1.0, 2.0, 3.0], [0.1, -0.5, 0.2], [-1.2, -0.8, 0.3]];
        let (grad_fd, laplac_fd) =
            grad_laplacian_finite_difference(&wave_function, &cfg, 1e-3).unwrap();

        assert!(grad_fd.all_close(&wave_function.gradient(&cfg).unwrap(), 1e-4));
        //assert_eq!(grad_fd, wave_function.gradient(&cfg).unwrap());
        assert!((laplac_fd - wave_function.laplacian(&cfg).unwrap()).abs() < 1e-4);
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

        assert!((wave_function.current_value().0 - value).abs() < 1e-14);
        assert!(wave_function.current_value().1.all_close(&grad, 1e-14));
        assert!((wave_function.current_value().2 - laplacian).abs() < 1e-14);
    }

    #[test]
    fn jastrow_slater_enqueue_test_fd() {
        use crate::util::grad_laplacian_finite_difference;

        let basis = GaussianBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0, 2.0, 3.0]);
        let orbitals = vec![
            Orbital::new(array![[1.0, 0.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 1.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 0.0, 1.0]], basis.clone()),
        ];

        let mut wave_function = JastrowSlater::new(array![1.0], orbitals, 0.1, 2);

        let mut cfg = array![[1.0, 2.0, 3.0], [0.1, -0.5, 0.2], [-1.2, -0.8, 0.3]];

        wave_function.refresh(&cfg);

        cfg[[0, 1]] = -1.0;

        let (grad, laplacian) =
            grad_laplacian_finite_difference(&wave_function, &cfg, 1e-3).unwrap();

        wave_function.enqueue_update(0, &cfg);

        assert!(wave_function
            .enqueued_value()
            .1
            .unwrap()
            .all_close(&grad, 1e-10));
        assert!((wave_function.enqueued_value().2.unwrap() - laplacian).abs() < 1e-8);

        assert!(!(wave_function.current_value().1.all_close(&grad, 1e-10)));
        assert!(!((wave_function.current_value().2 - laplacian).abs() < 1e-8));
    }
}
