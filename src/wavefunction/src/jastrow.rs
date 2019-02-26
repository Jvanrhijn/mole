use crate::error::Error;
use crate::traits::{Cache, Differentiate, Function};
use ndarray::{Array, Array1, Array2, Ix2};
use ndarray_linalg::Norm;
use std::collections::VecDeque;

type Vgl = (f64, Array2<f64>, f64);
type Ovgl = (Option<f64>, Option<Array2<f64>>, Option<f64>);

// f_ee in notes
// full Jastrow is then exp(f_ee)
struct ElectronElectronTerm {
    parms: Array1<f64>,
    scal: f64,
    num_up: usize,
}

#[allow(dead_code)]
impl ElectronElectronTerm {
    pub fn new(parms: Array1<f64>, scal: f64, num_up: usize) -> Self {
        // first two parameters are required
        assert!(parms.len() >= 1);
        if parms.len() > 1 {
            unimplemented!("Polynomial coefficients in fee are not yet implemented");
        }
        Self {
            parms,
            scal,
            num_up,
        }
    }
}

impl Function<f64> for ElectronElectronTerm {
    type D = Ix2;

    fn value(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        let num_elec = cfg.shape()[0];
        let nparms = self.parms.len();
        let mut value = 0.0;
        for i in 0..num_elec {
            for j in i + 1..num_elec {
                let b2 = if (i < self.num_up && j >= self.num_up)
                    || (i >= self.num_up && j < self.num_up)
                {
                    0.5
                } else {
                    0.25
                };
                let rij: f64 = (&cfg.slice(s![i, ..]) - &cfg.slice(s![j, ..])).norm_l2();
                let rij_scal = (1.0 - (-self.scal * rij).exp()) / self.scal;
                value += self.parms[0] * rij_scal / (1.0 + b2 * rij_scal);
                value += izip!(self.parms.slice(s![1..nparms]), 1..nparms)
                    .map(|(b, p)| b * rij_scal.powi(p as i32))
                    .sum::<f64>();
            }
        }
        Ok(value)
    }
}

impl Differentiate for ElectronElectronTerm {
    type D = Ix2;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Result<Array<f64, Self::D>, Error> {
        let nelec = cfg.shape()[0];
        let mut grad = Array2::<f64>::zeros((nelec, 3));
        for k in 0..nelec {
            let xk = cfg.slice(s![k, ..]);
            let mut grad_k = Array1::<f64>::zeros(3);
            for l in 0..nelec {
                if l == k {
                    continue;
                }
                let b2 = if (k < self.num_up && l >= self.num_up)
                    || (k >= self.num_up && l < self.num_up)
                {
                    0.5
                } else {
                    0.25
                };
                let xl = cfg.slice(s![l, ..]);
                let xkl = &xk - &xl;
                let rkl = xkl.norm_l2();
                let rkl_scal = (1.0 - (-self.scal * rkl).exp()) / self.scal;
                let magnitude =
                    self.parms[0] / (1.0 + b2 * rkl_scal).powi(2) * (-self.scal * rkl).exp();
                //let magnitude = (self.parms[0] / (1.0 + b2 * rkl_scal).powi(2)
                //    + izip!(1..nparms, self.parms.slice(s![1..nparms]))
                //        .map(|(p, b)| (p as f64) * b * rkl_scal.powi(p as i32 - 1))
                //        .sum::<f64>())
                //    * (-self.scal * rkl).exp();
                grad_k += &(magnitude * &xkl / rkl);
            }
            for i in 0..3 {
                grad[[k, i]] = grad_k[i];
            }
        }
        Ok(grad)
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        let mut laplacian = 0.0;
        let nelec = cfg.shape()[0];
        for k in 0..nelec {
            for l in 0..nelec {
                if k == l {
                    continue;
                }
                let b2 = if (k < self.num_up && l >= self.num_up)
                    || (k >= self.num_up && l < self.num_up)
                {
                    0.5
                } else {
                    0.25
                };
                let rkl: f64 = (&cfg.slice(s![k, ..]) - &cfg.slice(s![l, ..])).norm_l2();
                let exp = (-self.scal * rkl).exp();
                let rkl_scal = (1.0 - exp) / self.scal;

                let frac = self.parms[0] / (1.0 + b2 * rkl_scal).powi(2);
                let frac_2 = 2.0 * self.parms[0] * b2 / (1.0 + b2 * rkl_scal).powi(3);
                laplacian += 2.0 / rkl * frac * exp - exp.powi(2) * frac_2 - self.scal * exp * frac;
            }
        }
        Ok(laplacian)
    }
}

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

    fn value(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        Ok(self.fee.value(cfg)?.exp())
    }
}

// TODO Figure out a way to avoid computing the value and gradient repeatedly
impl Differentiate for JastrowFactor {
    type D = Ix2;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Result<Array2<f64>, Error> {
        let value = self.value(cfg)?;
        Ok(self.fee.gradient(cfg)? * value)
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        let value = self.value(cfg)?;
        Ok(value * (self.fee.laplacian(cfg)? + self.fee.gradient(cfg)?.norm_l2().powi(2)))
    }
}

impl Cache for JastrowFactor {
    type U = usize;

    fn refresh(&mut self, cfg: &Array2<f64>) {
        *self.value_queue.front_mut().unwrap() =
            self.value(cfg).expect("Failed to compute Jastrow value");
        *self.grad_queue.front_mut().unwrap() =
            self.gradient(cfg).expect("Failed to take Jastrow gradient");
        *self.laplac_queue.front_mut().unwrap() = self
            .laplacian(cfg)
            .expect("Failed to compute Jastrow Laplacian");
    }

    fn enqueue_update(&mut self, _ud: Self::U, cfg: &Array2<f64>) {
        self.value_queue
            .push_back(self.value(cfg).expect("Failed to store Jastrow value"));
        self.grad_queue.push_back(
            self.gradient(cfg)
                .expect("Failed to store Jastrow gradient"),
        );
        self.laplac_queue
            .push_back(self.laplacian(cfg).expect("Failed to store Laplacian"));
    }

    fn push_update(&mut self) {
        self.value_queue.pop_front();
        self.grad_queue.pop_front();
        self.laplac_queue.pop_front();
    }

    fn flush_update(&mut self) {
        self.value_queue.pop_back();
        self.grad_queue.pop_back();
        self.laplac_queue.pop_back();
    }

    fn current_value(&self) -> Vgl {
        match (
            self.value_queue.front(),
            self.grad_queue.front(),
            self.laplac_queue.front(),
        ) {
            (Some(&v), Some(g), Some(&l)) => (v, g.clone(), l),
            _ => panic!("Attempt to retrieve value from empty queue"),
        }
    }

    fn enqueued_value(&self) -> Ovgl {
        (
            self.value_queue
                .back()
                .and(Some(*self.value_queue.back().unwrap())),
            self.grad_queue
                .back()
                .and(Some(self.grad_queue.back().unwrap().clone())),
            self.laplac_queue
                .back()
                .and(Some(*self.laplac_queue.back().unwrap())),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::grad_laplacian_finite_difference;

    #[test]
    fn test_jastrow_factor() {
        let jas_ee = JastrowFactor::new(array![1.0], 1, 0.1, 1);
        let cfg = array![[1., -2., 3.], [4., 5., 6.], [-5., 8., -3.]];

        let (grad_fd, laplac_fd) = grad_laplacian_finite_difference(&jas_ee, &cfg, 1e-4).unwrap();
        assert!(grad_fd.all_close(&jas_ee.gradient(&cfg).unwrap(), 1e-8));
        assert!((laplac_fd - jas_ee.laplacian(&cfg).unwrap()).abs() < 1e-4);
    }

}
