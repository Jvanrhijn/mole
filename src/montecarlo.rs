// Standard imports
use std::vec::Vec;
//Third party imports
use ndarray::{Ix2, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
// First party imports
use traits::mcsamplers::*;
use traits::wavefunction::WaveFunction;
use traits::function::Function;
use traits::metropolis::Metropolis;
use traits::operator::Operator;
use error::Error;


pub struct Sampler<'a, T, V, U>
where T: Function<<U as Operator<T>>::V, D=Ix2> + WaveFunction,
      V: Metropolis<T>,
      U: Operator<T, V=f64>
{
    wave_function: &'a mut T,
    config: Array2<f64>,
    metropolis: V,
    observables: Vec<U>,
}

impl<'a, T, V, U> Sampler<'a, T, V, U>
where T: Function<<U as Operator<T>>::V, D=Ix2> + WaveFunction,
      V: Metropolis<T>,
      U: Operator<T, V=f64>
{
    pub fn new(wave_function: &'a mut T, mut metrop: V) -> Self {
        let nelec = wave_function.num_electrons();
        let cfg = Array2::<f64>::random((nelec, 3), Range::new(-1., 1.));
        *metrop.wf_val_prev_mut() = wave_function.value(&cfg)
            .expect("Failed to evaluate wave function");
        Self{
            wave_function,
            config: cfg,
            metropolis: metrop,
            observables: Vec::<U>::new(),
        }
    }

    pub fn add_observable(&mut self, operator: U) {
        self.observables.push(operator);
    }
}

impl<'a, T, V, U> MonteCarloSampler for Sampler<'a, T, V, U>
where T: Function<<U as Operator<T>>::V, D=Ix2> + WaveFunction,
      V: Metropolis<T>,
      U: Operator<T, V=f64>
{
    fn sample(&self) -> Result<Vec<f64>, Error> {
        Ok(self.observables.iter().map(|x| x.act_on(self.wave_function, &self.config)
            .expect("Failed to act on wave function with operator")).collect())
    }

    fn move_state(&mut self, elec: usize) {
        if let Some(config) = self.metropolis.move_state(self.wave_function, &self.config, elec) {
            self.config = config;
        }
    }

    fn num_electrons(&self) -> usize {
        self.wave_function.num_electrons()
    }

    fn num_observables(&self) -> usize {
        self.observables.len()
    }

}

pub struct Runner<S: MonteCarloSampler> {
    sampler: S,
    means: Vec<f64>,
    variances: Vec<f64>
}

impl<S> Runner<S>
where S: MonteCarloSampler
{
    pub fn new(sampler: S) -> Self {
        let mut means = Vec::<f64>::new();
        let mut variances = Vec::<f64>::new();
        means.resize(sampler.num_observables(), 0.0);
        variances.resize(sampler.num_observables(), 0.0);
        Self{sampler, means, variances}
    }

    pub fn run(&mut self, iters: usize) {
        let nelec = self.sampler.num_electrons();
        let mut count = 0_usize;
        for _ in 0..iters {
            for e in 0..nelec {
                self.sampler.move_state(e);
                let samples = self.sampler.sample().expect("Failed to sample observables");

                // calculating running mean and variance
                let means_prev = self.means.clone();
                self.means.iter_mut().zip(samples.iter()).for_each(|(x, y)| {
                    *x = (count as f64 * *x + y)/(count as f64 + 1.0)
                });
                self.variances.iter_mut().zip(samples.iter()).zip(self.means.iter()).zip(means_prev.iter())
                    .for_each(|(((v, x), m), mprev)| {
                        *v = *v + ((x - mprev)*(x - m) - *v)/(count as f64 + 1.0)
                });
                count += 1;
            }
        }
    }

    pub fn means(&self) -> &Vec<f64> {
        &self.means
    }

    pub fn variances(&self) -> &Vec<f64> {
        &self.variances
    }

}