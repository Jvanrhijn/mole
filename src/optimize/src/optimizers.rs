use crate::traits::*;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::SolveH;

#[derive(Clone)]
pub struct SteepestDescent {
    step_size: f64,
}

impl SteepestDescent {
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }
}

impl Optimizer for SteepestDescent {
    fn compute_parameter_update(&mut self, (energy_grad, _, _): &(Array1<f64>, Array1<f64>, Array2<f64>)) -> Array1<f64> {
        -&(self.step_size * energy_grad)
    }
}

#[derive(Clone)]
pub struct MomentumDescent {
    step_size: f64,
    momentum_parameter: f64,
    momentum: Array1<f64>,
}

impl MomentumDescent {
    pub fn new(step_size: f64, momentum_parameter: f64, nparm: usize) -> Self {
        Self {
            step_size, momentum_parameter, momentum: Array1::zeros(nparm)
        }
    }
}

impl Optimizer for MomentumDescent {
    fn compute_parameter_update(&mut self, (energy_grad, _, _): &(Array1<f64>, Array1<f64>, Array2<f64>)) -> Array1<f64> {
        self.momentum -= &(self.step_size * energy_grad);
        self.momentum_parameter * &self.momentum
    }
}

#[derive(Clone)]
pub struct StochasticReconfiguration {
    step_size: f64,
}

impl StochasticReconfiguration {
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }

    fn construct_sr_matrix(parm_grad: &Array2<f64>, wf_values: &Array1<f64>) -> Array2<f64> {
        let shape = parm_grad.shape();
        let nsamples = shape[0];
        let nparm = shape[1];

        // construct the stochastic reconfiguration matrix
        let mut sr_mat = Array2::<f64>::zeros((nparm, nparm));

        // build array2 of o_i values
        let mut sr_o = Array2::<f64>::zeros((nsamples, nparm));
        for n in 0..nsamples {
            for i in 0..nparm {
                sr_o[[n, i]] = parm_grad[[n, i]] / wf_values[n];
            }
        }

        // add the <Ok Ol> term to sr_mat
        for n in 0..nsamples {
            sr_mat += &(outer_product(
                &sr_o.slice(s![n, ..]).to_owned(),
                &sr_o.slice(s![n, ..]).to_owned(),
            ) / nsamples as f64);
        }

        let sr_o_avg = sr_o.mean_axis(Axis(0));

        // subtract <Ok><Ol>
        for i in 0..nparm {
            for j in 0..nparm {
                sr_mat -= sr_o_avg[i] * sr_o_avg[j];
            }
        }

        sr_mat //- &sr_o_avg_mat2
    }
}

impl Optimizer for StochasticReconfiguration {
    fn compute_parameter_update(
        &mut self,
        (energy_grad, wf_values, grad_parm): &(Array1<f64>, Array1<f64>, Array2<f64>),
    ) -> Array1<f64> {
        let sr_matrix = StochasticReconfiguration::construct_sr_matrix(grad_parm, wf_values);
        self.step_size * sr_matrix.solveh_into(-0.5 * energy_grad).unwrap()
    }
}

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let mut result = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}
