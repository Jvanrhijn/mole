use ndarray::{Array1, Array2, Array, Ix2};
use ndarray_linalg::Norm;
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
                let rij: f64 = (&cfg.slice(s![i, ..]) - &cfg.slice(s![j, ..])).norm_l2();
                value += self.parms[0]*rij/(1.0 + self.parms[1]*rij);
                let rij_power_series: Array1<_> = (1..nparms).map(|p| rij.powi(p as i32))
                    .collect();
                let innermost_sum: f64 = self.parms.slice(s![2..nparms])
                    .iter()
                    .zip(rij_power_series.iter())
                    .map(|(b, rij_p)| b*rij_p)
                    .sum();
                value += innermost_sum;
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
            let mut slice = grad.slice_mut(s![k, ..]);
            slice = {
                let mut magnitude = 0.0;
                for i in 0..nelec {
                    let rik: f64 = (&cfg.slice(s![i, ..]) - &cfg.slice(s![k, ..])).norm_l2();
                    magnitude += self.parms[0]/(1.0 + self.parms[1]*rik).powi(2);
                    let p = Array1::range(2.0, nparms as f64 + 1.0, 1.0);
                    magnitude += (&self.parms.slice(s![1..nparms]) * &p * p.map(|p| rik.powf(*p))).sum();
                }
                Array1::from_elem(3, magnitude).view_mut()
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
                for k in 0..nelec {
                    if i == k {
                        continue;
                    }
                    let rik: f64 = (&cfg.slice(s![i, ..]) - &cfg.slice(s![k, ..])).norm_l2();
                    laplacian += -2.0*self.parms[0]*self.parms[1]/(1.0 + self.parms[1] * rik).powi(3);
                    for p in 2..nparm - 1 {
                        let p = p as f64;
                        laplacian += self.parms[p as usize] * p * (p - 1.0) * rik.powf(p - 3.0);
                    }
                }
            }
            laplacian *= xk.sum();
        }
        Ok(laplacian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jastrow_ee() {
        let jas_ee = ElectronElectronTerm::new(array![1.0, 1.0, 1.0]);
        let cfg = array![[0.0, 0.0, 1.0]];
        let value = jas_ee.value(&cfg);
        assert_eq!(value.unwrap(), 0.0);
        let grad = jas_ee.gradient(&cfg);
        assert_eq!(jas_ee.gradient(&cfg).unwrap(), Array2::zeros((1, 3)));
        let laplac = jas_ee.laplacian(&cfg);
        assert_eq!(laplac.unwrap(), 0.0)
    }
}
