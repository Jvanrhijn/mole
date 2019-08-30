use errors::Error::{self, EmptyCacheError};
use ndarray::{Array, Array1, Array2, Ix2};
use ndarray_linalg::Norm;
use optimize::Optimize;
use std::collections::VecDeque;
use wavefunction_traits::{Cache, Differentiate, Function};

type Vgl = (f64, Array2<f64>, f64);
type Ovgl = (Option<f64>, Option<Array2<f64>>, Option<f64>);

type Result<T> = std::result::Result<T, Error>;

// f_ee in notes
// full Jastrow is then exp(f_ee)
#[derive(Clone)]
struct ElectronElectronTerm {
    parms: Array1<f64>,
    scal: f64,
    num_up: usize,
}

#[allow(dead_code)]
impl ElectronElectronTerm {
    pub fn new(parms: Array1<f64>, scal: f64, num_up: usize) -> Self {
        assert!(!parms.is_empty());
        Self {
            parms,
            scal,
            num_up,
        }
    }

    fn get_b1(&self, i: usize, j: usize) -> f64 {
        if i < self.num_up && j >= self.num_up || i >= self.num_up && j < self.num_up {
            0.5
        } else {
            0.25
        }
    }
}

impl Function<f64> for ElectronElectronTerm {
    type D = Ix2;

    fn value(&self, cfg: &Array<f64, Self::D>) -> Result<f64> {
        let num_elec = cfg.shape()[0];
        let nparm = self.parms.len();
        let mut value = 0.0;
        for i in 0..num_elec {
            for j in i + 1..num_elec {
                let b1 = self.get_b1(i, j);
                let rij: f64 = (&cfg.slice(s![i, ..]) - &cfg.slice(s![j, ..])).norm_l2();
                let rij_scal = (1.0 - (-self.scal * rij).exp()) / self.scal;
                value += b1 * rij_scal / (1.0 + self.parms[0] * rij_scal);
                if nparm > 1 {
                    value += izip!(2..=nparm, self.parms.slice(s![1..]))
                        .map(|(p, b)| b * rij_scal.powi(p as i32))
                        .sum::<f64>();
                }
            }
        }
        Ok(value)
    }
}

impl Differentiate for ElectronElectronTerm {
    type D = Ix2;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Result<Array<f64, Self::D>> {
        let nelec = cfg.shape()[0];
        let nparm = self.parms.len();
        let mut grad = Array2::<f64>::zeros((nelec, 3));
        for k in 0..nelec {
            let xk = cfg.slice(s![k, ..]);
            let mut grad_k = Array1::<f64>::zeros(3);
            for l in 0..nelec {
                if l == k {
                    continue;
                }
                let b1 = self.get_b1(k, l);
                let xl = cfg.slice(s![l, ..]);
                let xkl = &xk - &xl;
                let rkl = xkl.norm_l2();
                let exp = (-self.scal * rkl).exp();
                let rkl_scal = (1.0 - exp) / self.scal;
                let magnitude = b1 / (1.0 + self.parms[0] * rkl_scal).powi(2) * exp
                    + if nparm > 1 {
                        izip!(2..=nparm, self.parms.slice(s![1..]))
                            .map(|(p, b)| b * p as f64 * exp * rkl_scal.powi(p as i32 - 1))
                            .sum::<f64>()
                    } else {
                        0.0
                    };
                grad_k += &(magnitude * &xkl / rkl);
            }
            let mut slice = grad.slice_mut(s![k, ..]);
            slice += &grad_k;
        }
        Ok(grad)
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64> {
        let mut laplacian = 0.0;
        let nparm = self.parms.len();
        let nelec = cfg.shape()[0];
        for k in 0..nelec {
            for l in 0..nelec {
                if k == l {
                    continue;
                }
                let b1 = self.get_b1(k, l);
                let rkl: f64 = (&cfg.slice(s![k, ..]) - &cfg.slice(s![l, ..])).norm_l2();
                let exp = (-self.scal * rkl).exp();
                let rkl_scal = (1.0 - exp) / self.scal;

                let frac = b1 / (1.0 + self.parms[0] * rkl_scal).powi(2);
                let frac_2 = 2.0 * b1 * self.parms[0] / (1.0 + self.parms[0] * rkl_scal).powi(3);
                laplacian += 2.0 / rkl * frac * exp - exp.powi(2) * frac_2 - self.scal * exp * frac;
                laplacian += exp.powi(2)
                    * izip!(2..=nparm, self.parms.slice(s![1..]))
                        .map(|(p, b)| (p * (p - 1)) as f64 * b * rkl_scal.powi(p as i32 - 2))
                        .sum::<f64>();
                laplacian += (2.0 / rkl - self.scal)
                    * exp
                    * izip!(2..=nparm, self.parms.slice(s![1..]))
                        .map(|(p, b)| (p as f64) * b * rkl_scal.powi(p as i32 - 1))
                        .sum::<f64>();
            }
        }
        Ok(laplacian)
    }
}

impl Optimize for ElectronElectronTerm {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>> {
        let num_elec = cfg.shape()[0];
        let mut grad_bp = Array1::<f64>::zeros(self.parms.len());
        for i in 0..num_elec {
            for j in i + 1..num_elec {
                let b1 = self.get_b1(i, j);
                let rij: f64 = (&cfg.slice(s![i, ..]) - &cfg.slice(s![j, ..])).norm_l2();
                let rij_scal = (1.0 - (-self.scal * rij).exp()) / self.scal;
                grad_bp[0] -= -b1 * rij_scal.powi(2) * (1.0 + self.parms[0] * rij_scal).powi(-2);
                let mut grad_rest = grad_bp.slice_mut(s![1..]);
                grad_rest -= &(3..=self.parms.len() + 1)
                    .map(|p| rij_scal.powi(p as i32 - 1))
                    .collect::<Array1<f64>>();
            }
        }
        Ok(grad_bp)
    }

    fn update_parameters(&mut self, deltap: &Array1<f64>) {
        self.parms += deltap;
    }

    fn parameters(&self) -> &Array1<f64> {
        &self.parms
    }

    fn num_parameters(&self) -> usize {
        self.parms.len()
    }
}

#[derive(Clone)]
pub struct JastrowFactor {
    fee: ElectronElectronTerm,
    value_queue: VecDeque<f64>,
    grad_queue: VecDeque<Array2<f64>>,
    laplac_queue: VecDeque<f64>,
}

impl JastrowFactor {
    pub fn new(parms: Array1<f64>, num_electrons: usize, scal: f64, num_up: usize) -> Self {
        let value_queue = VecDeque::from(vec![0.0]);
        let grad_queue = VecDeque::from(vec![Array2::ones((num_electrons, 3))]);
        let laplac_queue = VecDeque::from(vec![0.0]);
        Self {
            fee: ElectronElectronTerm::new(parms, scal, num_up),
            value_queue,
            grad_queue,
            laplac_queue,
        }
    }
}

impl Function<f64> for JastrowFactor {
    type D = Ix2;

    fn value(&self, cfg: &Array<f64, Self::D>) -> Result<f64> {
        Ok(self.fee.value(cfg)?.exp())
    }
}

// TODO Figure out a way to avoid computing the value and gradient repeatedly
impl Differentiate for JastrowFactor {
    type D = Ix2;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Result<Array2<f64>> {
        let value = self.value(cfg)?;
        Ok(self.fee.gradient(cfg)? * value)
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64> {
        let value = self.value(cfg)?;
        Ok(value * (self.fee.laplacian(cfg)? + self.fee.gradient(cfg)?.norm_l2().powi(2)))
    }
}

impl Optimize for JastrowFactor {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(self.fee.parameter_gradient(cfg)? * self.current_value()?.0)
    }

    fn update_parameters(&mut self, deltap: &Array1<f64>) {
        self.fee.update_parameters(deltap);
    }

    fn parameters(&self) -> &Array1<f64> {
        self.fee.parameters()
    }

    fn num_parameters(&self) -> usize {
        self.fee.num_parameters()
    }
}

impl Cache for JastrowFactor {
    type U = usize;

    fn refresh(&mut self, cfg: &Array2<f64>) -> Result<()> {
        *self.value_queue.front_mut().ok_or(EmptyCacheError)? = self.value(cfg)?;
        *self.grad_queue.front_mut().ok_or(EmptyCacheError)? = self.gradient(cfg)?;
        *self.laplac_queue.front_mut().ok_or(EmptyCacheError)? = self.laplacian(cfg)?;
        self.flush_update();
        Ok(())
    }

    fn enqueue_update(&mut self, _ud: Self::U, cfg: &Array2<f64>) -> Result<()> {
        self.value_queue.push_back(self.value(cfg)?);
        self.grad_queue.push_back(self.gradient(cfg)?);
        self.laplac_queue.push_back(self.laplacian(cfg)?);
        Ok(())
    }

    fn push_update(&mut self) {
        self.value_queue.pop_front();
        self.grad_queue.pop_front();
        self.laplac_queue.pop_front();
    }

    fn flush_update(&mut self) {
        if self.value_queue.len() == 2 {
            self.value_queue.pop_back();
        }
        if self.grad_queue.len() == 2 {
            self.grad_queue.pop_back();
        }
        if self.laplac_queue.len() == 2 {
            self.laplac_queue.pop_back();
        }
    }

    fn current_value(&self) -> Result<Vgl> {
        Ok((
            *self.value_queue.front().ok_or(EmptyCacheError)?,
            self.grad_queue.front().ok_or(EmptyCacheError)?.clone(),
            *self.laplac_queue.front().ok_or(EmptyCacheError)?,
        ))
    }

    fn enqueued_value(&self) -> Ovgl {
        (
            self.value_queue.back().copied(),
            self.grad_queue.back().cloned(),
            self.laplac_queue.back().copied(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::grad_laplacian_finite_difference;
    use ndarray_rand::RandomExt;
    use rand::{distributions::Range, SeedableRng, StdRng};

    #[test]
    fn test_jastrow_factor() {
        const NUM_ELECTRONS: usize = 4;
        const NUM_TESTS: usize = 100;

        let jas_ee = JastrowFactor::new(
            array![0.5, 0.1, 0.01],
            NUM_ELECTRONS,
            0.1,
            NUM_ELECTRONS / 2,
        );
        let mut rng = StdRng::from_seed([0; 32]);

        for _ in 0..NUM_TESTS {
            // generate random configuration
            let cfg = Array2::<f64>::random_using(
                (NUM_ELECTRONS, 3),
                Range::new(-1.0_f64, 1.0_f64),
                &mut rng,
            );
            let (grad_fd, laplac_fd) =
                grad_laplacian_finite_difference(&jas_ee, &cfg, 1e-3).unwrap();
            assert!(grad_fd.all_close(&jas_ee.gradient(&cfg).unwrap(), 1e-4));
            assert!((laplac_fd - jas_ee.laplacian(&cfg).unwrap()).abs() < 1e-4);
        }
    }
}
