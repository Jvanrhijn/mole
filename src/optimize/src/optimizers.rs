use crate::traits::*;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::SolveH;
use std::collections::VecDeque;

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
    fn compute_parameter_update(
        &mut self,
        (energy_grad, _, _, _): &(Array1<f64>, Array1<f64>, Array2<f64>, Array1<f64>),
    ) -> Array1<f64> {
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
            step_size,
            momentum_parameter,
            momentum: Array1::zeros(nparm),
        }
    }
}

impl Optimizer for MomentumDescent {
    fn compute_parameter_update(
        &mut self,
        (energy_grad, _, _, _): &(Array1<f64>, Array1<f64>, Array2<f64>, Array1<f64>),
    ) -> Array1<f64> {
        self.momentum -= &(self.step_size * energy_grad);
        self.momentum_parameter * &self.momentum
    }
}

#[derive(Clone)]
pub struct NesterovMomentum {
    step_size: f64,
    momentum_parameter: f64,
    momentum: Array1<f64>,
    momentum_prev: Array1<f64>,
}

impl NesterovMomentum {
    pub fn new(step_size: f64, momentum_parameter: f64, nparm: usize) -> Self {
        Self {
            step_size: -step_size,
            momentum_parameter,
            momentum: Array1::zeros(nparm),
            momentum_prev: Array1::zeros(nparm),
        }
    }
}

impl Optimizer for NesterovMomentum {
    fn compute_parameter_update(
        &mut self,
        (energy_grad, _, _, _): &(Array1<f64>, Array1<f64>, Array2<f64>, Array1<f64>),
    ) -> Array1<f64> {
        self.momentum_prev = self.momentum.clone();
        self.momentum = self.momentum_parameter * &self.momentum + self.step_size * energy_grad;
        -(self.momentum_parameter * &self.momentum_prev
            + (1.0 + self.momentum_parameter) * &self.momentum)
    }
}

#[derive(Clone)]
pub struct OnlineLbfgs {
    step_size: f64,
    history: usize,
    grad_prev: Array1<f64>,
    pars_prev: Array1<f64>,
    s: VecDeque<Array1<f64>>,
    y: VecDeque<Array1<f64>>,
    iter: usize,
}

impl OnlineLbfgs {
    pub fn new(step_size: f64, history: usize, nparm: usize) -> Self {
        const EPS: f64 = 1e-5;
        Self {
            step_size,
            history,
            grad_prev: Array1::from_elem(nparm, EPS),
            pars_prev: Array1::from_elem(nparm, EPS),
            s: VecDeque::with_capacity(history),
            y: VecDeque::with_capacity(history),
            iter: 0,
        }
    }

    fn initial_direction(&self, gradient: &Array1<f64>) -> Array1<f64> {
        // damping parameter for search direction scaling
        const EPS: f64 = 1e-10;

        let mut p = -gradient;
        let mut alphas = Vec::new();

        // first of two-loop recursion
        for (s, y) in self.s.iter().rev().zip(self.y.iter().rev()) {
            let alpha = s.dot(&p) / s.dot(y);
            p -= &(alpha * y);
            alphas.push(alpha);
        }
        // scale search direction by averaging
        if self.iter == 0 {
            p *= EPS
        } else {
            let mut tot = 0.0;
            for (s, y) in self.s.iter().zip(self.y.iter()) {
                tot += s.dot(y) / y.dot(y);
            }
            p *= tot / (usize::min(self.iter, self.history)) as f64;
        }
        // second of two-loop recursion
        for (alpha, s, y) in izip!(alphas.iter().rev(), self.s.iter(), self.y.iter()) {
            let beta = y.dot(&p) / y.dot(s);
            p += &((alpha - beta) * s);
        }
        // p is now the search direction
        p
    }

    fn update_curvature_pairs(&mut self, pars: &Array1<f64>, grad: &Array1<f64>) {
        let s = pars - &self.pars_prev;
        let y = grad - &self.grad_prev;
        if self.s.len() >= self.history {
            self.s.pop_front();
        } 
        self.s.push_back(s);
        if self.y.len() >= self.history {
            self.y.pop_front();
        }
        self.y.push_back(y);
    }
}

impl Optimizer for OnlineLbfgs {
    fn compute_parameter_update(
        &mut self,
        (energy_grad, _, _, pars): &(Array1<f64>, Array1<f64>, Array2<f64>, Array1<f64>),
    ) -> Array1<f64> {
        self.update_curvature_pairs(&pars, &energy_grad);
        let p = self.initial_direction(energy_grad);
        let s = -self.step_size * p;
        self.update_curvature_pairs(&pars, &energy_grad);
        self.grad_prev = energy_grad.clone();
        self.pars_prev = pars.clone();
        self.iter += 1;
        s
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
        (energy_grad, wf_values, grad_parm, _): &(
            Array1<f64>,
            Array1<f64>,
            Array2<f64>,
            Array1<f64>,
        ),
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
