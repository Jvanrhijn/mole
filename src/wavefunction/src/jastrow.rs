use crate::error::Error;
use crate::traits::{Cache, Differentiate, Function};
use ndarray::{Array, Array1, Array2, Ix2};
use ndarray_linalg::Norm;
use std::collections::VecDeque;

// f_ee in notes
// full Jastrow is then exp(f_ee + f_en + f_een)
struct ElectronElectronTerm {
    parms: Array1<f64>,
    scal: f64,
}

#[allow(dead_code)]
impl ElectronElectronTerm {
    pub fn new(parms: Array1<f64>, scal: f64) -> Self {
        // first two parameters are required
        assert!(parms.len() >= 2);
        if parms.len() > 2 {
            unimplemented!("Polynomial coefficients in fee are not yet implemented");
        }
        Self { parms, scal }
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
                let rij: f64 = (&cfg.slice(s![i, ..]) - &cfg.slice(s![j, ..])).norm_l2();
                let rij_scal = (1.0 - (-self.scal * rij).exp()) / self.scal;
                value += self.parms[0] * rij_scal / (1.0 + self.parms[1] * rij_scal);
                value += izip!(self.parms.slice(s![2..nparms]), 2..nparms)
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
        let nparms = self.parms.len();
        let nelec = cfg.shape()[0];
        let mut grad = Array2::<f64>::zeros((nelec, 3));
        for k in 0..nelec {
            let xk = cfg.slice(s![k, ..]);
            let mut grad_k = Array1::<f64>::zeros(3);
            for l in 0..nelec {
                if l == k {
                    continue;
                }
                let xl = cfg.slice(s![l, ..]);
                let xkl = (&xk - &xl);
                let rkl = xkl.norm_l2();
                let rkl_scal = (1.0 - (-self.scal * rkl).exp()) / self.scal;
                let magnitude = -(self.parms[0] / (1.0 + self.parms[1] * rkl_scal).powi(2)
                    + izip!(2..nparms, self.parms.slice(s![2..nparms]))
                        .map(|(p, b)| (p as f64) * b * rkl_scal.powi(p as i32 - 1))
                        .sum::<f64>())
                    * (-self.scal * rkl).exp();
                grad_k += &(magnitude * &xkl / rkl);
            }
            for i in 0..3 {
                grad[[k, i]] = grad_k[i];
            }
        }
        Ok(0.5 * grad)
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        let mut laplacian = 0.0;
        let nelec = cfg.shape()[0];
        let nparm = self.parms.len();
        for k in 0..nelec {
            let xk = cfg.slice(s![k, ..]);
            for l in 0..nelec {
                if k == l {
                    continue;
                }
                let xl = cfg.slice(s![l, ..]);
                let rkl: f64 = (&cfg.slice(s![k, ..]) - &cfg.slice(s![l, ..])).norm_l2();
                let exp = (-self.scal * rkl).exp();
                let rkl_scal = (1.0 - exp) / self.scal;
                let g = exp
                    * (self.parms[0] / (1.0 + self.parms[1] * rkl_scal).powi(2)
                        + izip!(2..nparm, self.parms.slice(s![2..]))
                            .map(|(p, b)| (p as f64) * b * rkl_scal.powi(p as i32 - 1))
                            .sum::<f64>());
                let inner_lapl = -2.0 * self.parms[0] / (1.0 + self.parms[1]).powi(3)
                    + izip!(2..nparm, self.parms.slice(s![2..]))
                        .map(|(p, b)| (p * (p - 1)) as f64 * b * rkl_scal.powi(p as i32 - 2))
                        .sum::<f64>();
                laplacian += g * (2.0 / rkl - self.scal) + self.scal * exp.powi(2) * inner_lapl;
            }
        }
        Ok(0.5 * laplacian)
    }
}

pub struct JastrowFactor {
    fee: ElectronElectronTerm,
    value_queue: VecDeque<f64>,
    grad_queue: VecDeque<Array2<f64>>,
    laplac_queue: VecDeque<f64>,
}

impl JastrowFactor {
    pub fn new(parms: Array1<f64>, num_electrons: usize, scal: f64) -> Self {
        let value_queue = VecDeque::from(vec![0.0]);
        let grad_queue = VecDeque::from(vec![Array2::ones((num_electrons, 3))]);
        let laplac_queue = VecDeque::from(vec![0.0]);
        Self {
            fee: ElectronElectronTerm::new(parms, scal),
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
    use ndarray::Axis;
    use ndarray_linalg::Norm;
    const EPS: f64 = 1e-13;

    #[test]
    fn test_jastrow_ee() {
        let jas_ee = ElectronElectronTerm::new(array![1.0, 1.0], 1.0);
        let x1 = array![[1.0, 2.0, 3.0]];
        let x2 = array![[4.0, 5.0, 6.0]];
        let cfg = stack!(Axis(0), x1, x2);
        let x12 = &x1 - &x2;
        let r12: f64 = x12.norm_l2();
        let r12_scal = 1.0 - (-r12).exp();

        let value = jas_ee.value(&cfg);
        let value_exact = r12_scal / (1.0 + r12_scal);
        assert_eq!(value.unwrap(), value_exact);

        let grad = jas_ee.gradient(&cfg).unwrap();
        let grad_1 =
            -0.5 * (1.0 / (1.0 + r12_scal).powi(2)) * (&x12 / r12) * (-r12).exp();
        let grad_exact = stack![Axis(0), grad_1, -grad_1.clone()];
        assert!(grad.all_close(&grad_exact, EPS));

        let laplac = jas_ee.laplacian(&cfg);
        let laplac_exact = 0.5
            * (2.0
                * ((-r12).exp()
                    * (1.0 / (1.0 + r12_scal).powi(2) )
                    * (2.0 / r12 - 1.0)
                    + (-2.0 * r12).exp() * (- 2.0 / (1.0 + r12_scal).powi(3))));
        assert!((laplac.unwrap() - laplac_exact).abs() < 1e-4);
    }
}
