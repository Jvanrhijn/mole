use crate::error::Error;
use crate::traits::{Cache, Differentiate, Function};
use ndarray::{Array, Array1, Array2, Ix2};
use ndarray_linalg::Norm;
use std::collections::VecDeque;

// f_ee in notes
// full Jastrow is then exp(f_ee + f_en + f_een)
struct ElectronElectronTerm {
    parms: Array1<f64>,
}

#[allow(dead_code)]
impl ElectronElectronTerm {
    pub fn new(parms: Array1<f64>) -> Self {
        Self { parms }
    }
}

impl Function<f64> for ElectronElectronTerm {
    type D = Ix2;

    fn value(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        let num_elec = cfg.shape()[0];
        let nparms = self.parms.len();
        let mut value = 0.0;
        for i in 0..num_elec {
            for j in i..num_elec {
                if i == j {
                    continue;
                }
                let rij: f64 = (&cfg.slice(s![i, ..]) - &cfg.slice(s![j, ..])).norm_l2();
                value += self.parms[0] * rij / (1.0 + self.parms[1] * rij);
                value += izip!(self.parms.slice(s![2..nparms]), 2..nparms)
                    .map(|(b, p)| b * rij.powi(p as i32))
                    .sum::<f64>();
            }
        }
        Ok(value)
    }
}

impl Differentiate for ElectronElectronTerm {
    type D = Ix2;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Result<Array<f64, Self::D>, Error> {
        let nparms = self.parms.len();
        let nelec = cfg.shape()[0];
        let mut grad = Array2::<f64>::zeros((nelec, 3));
        for k in 0..nelec {
            let slice = {
                let mut magnitude = 0.0;
                let mut component = Array1::<f64>::zeros(3);
                for i in 0..nelec {
                    let separation = &cfg.slice(s![i, ..]) - &cfg.slice(s![k, ..]);
                    if i == k {
                        continue;
                    }
                    let rik: f64 = separation.norm_l2();
                    magnitude += self.parms[0] / (1.0 + self.parms[1] * rik).powi(2);
                    magnitude += izip!(self.parms.slice(s![2..nparms]), 2..nparms)
                        .map(|(b, p)| (p as f64) * b * rik.powi(p as i32 - 1))
                        .sum::<f64>();
                    component += &(separation * magnitude / rik);
                }
                component
            };
            for i in 0..3 {
                grad[[k, i]] = slice[i];
            }
        }
        Ok(grad)
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        let mut laplacian = 0.0;
        let nelec = cfg.shape()[0];
        let nparm = self.parms.len();
        for k in 0..nelec {
            let xk = cfg.slice(s![k, ..]);
            for i in 0..nelec {
                if i == k {
                    continue;
                }
                let xi = cfg.slice(s![i, ..]);
                let rik: f64 = (&cfg.slice(s![i, ..]) - &cfg.slice(s![k, ..])).norm_l2();
                laplacian += (-2.0 * self.parms[0] * self.parms[1]
                    / (1.0 + self.parms[1] * rik).powi(3)
                    + izip!(self.parms.slice(s![2..nparm]), 2..nparm)
                        .map(|(b, p)| b * (p * (p - 1)) as f64 * rik.powi(p as i32 - 3))
                        .sum::<f64>())
                    * (&xi - &xk).scalar_sum()
                    / rik;
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
    pub fn new(parms: Array1<f64>, num_electrons: usize) -> Self {
        let value_queue = VecDeque::from(vec![0.0]);
        let grad_queue = VecDeque::from(vec![Array2::ones((num_electrons, 3))]);
        let laplac_queue = VecDeque::from(vec![0.0]);
        Self {
            fee: ElectronElectronTerm::new(parms),
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

impl Cache<Array2<f64>> for JastrowFactor {
    type A = Array2<f64>;
    type V = (f64, Array2<f64>, f64);
    type OV = (Option<f64>, Option<Array2<f64>>, Option<f64>);
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

    fn enqueue_update(&mut self, ud: Self::U, cfg: &Array2<f64>) {
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

    fn current_value(&self) -> Self::V {
        match (
            self.value_queue.front(),
            self.grad_queue.front(),
            self.laplac_queue.front(),
        ) {
            (Some(&v), Some(g), Some(&l)) => (v, g.clone(), l),
            _ => panic!("Attempt to retrieve value from empty queue"),
        }
    }

    fn enqueued_value(&self) -> Self::OV {
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
    const EPS: f64 = 1e-13;

    #[test]
    fn test_jastrow_ee() {
        let jas_ee = ElectronElectronTerm::new(array![1.0, 2.0, 3.0]);
        let cfg = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let value = jas_ee.value(&cfg);
        let value_exact = 3.0 * 3.0_f64.sqrt() / (1.0 + 6.0 * 3.0_f64.sqrt()) + 3.0 * 27.0;
        assert_eq!(value.unwrap(), value_exact);

        let grad = jas_ee.gradient(&cfg).unwrap();
        let grad_1 =
            (1.0 / (1.0 + 6.0 * 3.0_f64.sqrt()).powi(2) + 18.0 * 3.0_f64.sqrt()) / 3.0_f64.sqrt();
        let grad_exact = array![[grad_1, grad_1, grad_1], [-grad_1, -grad_1, -grad_1]];
        assert!(grad.all_close(&grad_exact, EPS));

        let laplac = jas_ee.laplacian(&cfg);
        let laplac_exact = 0.0; // symmetry; $\Delta_1 f_ee = - \Delta_2 f_ee
        assert_eq!(laplac.unwrap(), laplac_exact);
    }
}
