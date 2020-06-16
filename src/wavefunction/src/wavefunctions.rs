// Defines various wave function representations, e.g. Jastrow-Slater
use std::collections::VecDeque;
// third party imports
use ndarray::{Array1, Array2, Axis, Ix2};
// first party imports
use crate::determinant::Slater;
use crate::jastrow::JastrowFactor;
use crate::orbitals::Orbital;
use basis::BasisSet;
use errors::Error;
use optimize::Optimize;
use wavefunction_traits::{Differentiate, Function, WaveFunction};

type Result<T> = std::result::Result<T, Error>;

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
    pub fn new(orbs: Vec<Orbital<T>>) -> Result<Self> {
        Ok(Self {
            det: Slater::new(orbs)?,
        })
    }
}

impl<T> Function<f64> for SingleDeterminant<T>
where
    T: BasisSet,
{
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64> {
        self.det.value(cfg)
    }
}

impl<T> Differentiate for SingleDeterminant<T>
where
    T: BasisSet,
{
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Result<Array2<f64>> {
        self.det.gradient(cfg)
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64> {
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
    pub fn new(mut orbitals: Vec<Orbital<T>>, num_up: usize) -> Result<Self> {
        let nelec = orbitals.len();
        let orb_up = orbitals.drain(0..num_up).collect();
        Ok(Self {
            det_up: Slater::new(orb_up)?,
            det_down: Slater::new(orbitals)?,
            num_up,
            value_cache: VecDeque::from(vec![1.0]),
            grad_cache: VecDeque::from(vec![Array2::zeros((nelec, 3))]),
            laplacian_cache: VecDeque::from(vec![1.0]),
        })
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

    fn value(&self, cfg: &Array2<f64>) -> Result<f64> {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        Ok(self.det_up.value(&cfg_up)? * self.det_down.value(&cfg_down)?)
    }
}

impl<T: BasisSet> Differentiate for SpinDeterminantProduct<T> {
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Result<Array2<f64>> {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        let det_up_v = self.det_up.value(&cfg_up)?;
        let det_down_v = self.det_down.value(&cfg_down)?;
        Ok(stack![
            Axis(0),
            det_down_v * self.det_up.gradient(&cfg_up)?,
            det_up_v * self.det_down.gradient(&cfg_down)?
        ])
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64> {
        let (cfg_up, cfg_down) = self.split_config(cfg);
        let det_up_v = self.det_up.value(&cfg_up)?;
        let det_down_v = self.det_down.value(&cfg_down)?;
        Ok(det_down_v * self.det_up.laplacian(&cfg_up)?
            + det_up_v * self.det_down.laplacian(&cfg_down)?)
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
    pub fn new(
        parms: Array1<f64>,
        orbitals: Vec<Orbital<T>>,
        scal: f64,
        num_up: usize,
    ) -> Result<Self> {
        let num_elec = orbitals.len();
        let jastrow = JastrowFactor::new(parms, num_elec, scal, num_up);
        let value_cache = VecDeque::from(vec![1.0]);
        let grad_cache = VecDeque::from(vec![Array2::<f64>::ones((num_elec, 3))]);
        let lapl_cache = VecDeque::from(vec![0.0]);
        Ok(Self {
            det: SpinDeterminantProduct::new(orbitals, num_up)?,
            jastrow,
            value_cache,
            grad_cache,
            lapl_cache,
        })
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

    fn value(&self, cfg: &Array2<f64>) -> Result<f64> {
        Ok(self.jastrow.value(cfg)? * self.det.value(cfg)?)
    }
}

impl<T: BasisSet> Differentiate for JastrowSlater<T> {
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Result<Array2<f64>> {
        Ok(self.jastrow.gradient(cfg)? * self.det.value(cfg)?
            + self.jastrow.value(cfg)? * self.det.gradient(cfg)?)
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64> {
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


impl<T: BasisSet> Optimize for JastrowSlater<T> {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>> {
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
    fn jastrow_slater_grad_laplacian() {
        use crate::util::grad_laplacian_finite_difference;
        let basis = GaussianBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0, 2.0, 3.0]);
        let orbitals = vec![
            Orbital::new(array![[1.0, 0.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 1.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 0.0, 1.0]], basis.clone()),
        ];
        let wave_function = JastrowSlater::new(array![1.0, 0.01, 0.01], orbitals, 0.1, 2).unwrap();

        let cfg = array![[1.0, 2.0, 3.0], [0.1, -0.5, 0.2], [-1.2, -0.8, 0.3]];
        let (grad_fd, laplac_fd) =
            grad_laplacian_finite_difference(&wave_function, &cfg, 1e-3).unwrap();

        assert!(grad_fd.all_close(&wave_function.gradient(&cfg).unwrap(), 1e-4));
        //assert_eq!(grad_fd, wave_function.gradient(&cfg).unwrap());
        assert!((laplac_fd - wave_function.laplacian(&cfg).unwrap()).abs() < 1e-4);
    }
}
