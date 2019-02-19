use ndarray::{Array1, Array2, Array, Ix2};
use ndarray_linalg::Norm;
use itertools;
use crate::traits::{Function, Differentiate, Cache};
use crate::error::Error;

// f_ee in notes
// full Jastrow is then exp(f_ee + f_en + f_een)
struct ElectronElectronTerm {
    parms: Array1<f64>
}

impl ElectronElectronTerm {
    pub fn new(parms: Array1<f64>) -> Self {
        Self{parms}
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
                value += self.parms[0]*rij/(1.0 + self.parms[1]*rij);
                value += izip!(self.parms.slice(s![2..nparms]), (2..nparms))
                    .map(|(b, p)| b*rij.powi(p as i32))
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
                let mut component = Array1::<f64>::zeros( 3);
                for i in 0..nelec {
                    let separation = &cfg.slice(s![i, ..]) - &cfg.slice(s![k, ..]);
                    if i == k {
                        continue;
                    }
                    let rik: f64 = separation.norm_l2();
                    magnitude += self.parms[0]/(1.0 + self.parms[1]*rik).powi(2);
                    magnitude += izip!(self.parms.slice(s![2..nparms]), (2..nparms))
                        .map(|(b, p)| (p as f64)*b*rik.powi(p as i32 - 1))
                        .sum::<f64>();
                    component += &(separation*magnitude/rik);
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
                laplacian += (-2.0*self.parms[0]*self.parms[1]/(1.0 + self.parms[1] * rik).powi(3)
                    + izip!(self.parms.slice(s![2..nparm]), (2..nparm))
                        .map(|(b, p)| b*(p*(p - 1)) as f64*rik.powi(p as i32 - 3))
                        .sum::<f64>())
                    * (&xi - &xk).sum()/rik;
            }
        }
        Ok(laplacian)
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
        let value_exact = 3.0*3.0_f64.sqrt()/(1.0 + 6.0*3.0_f64.sqrt()) + 3.0*27.0;
        assert_eq!(value.unwrap(), value_exact);

        let grad = jas_ee.gradient(&cfg).unwrap();
        let grad_1 = (1.0/(1.0 + 6.0*3.0_f64.sqrt()).powi(2) + 18.0*3.0_f64.sqrt())/3.0_f64.sqrt();
        let grad_exact = array![[grad_1, grad_1, grad_1], [-grad_1, -grad_1, -grad_1]];
        for i in 0..2 {
            for j in 0..3 {
                assert!((grad[[i, j]] - grad_exact[[i, j]]).abs() < EPS);
            }
        }

        let laplac = jas_ee.laplacian(&cfg);
        let laplac_exact = 0.0; // symmetry; $\Delta_1 f_ee = - \Delta_2 f_ee
        assert_eq!(laplac.unwrap(),  laplac_exact);
    }
}
