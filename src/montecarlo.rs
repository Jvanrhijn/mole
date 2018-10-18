// Standard imports
use std::vec::Vec;
//Third party imports
use ndarray::{Ix2, Array1, Array2, Axis};
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
    pub fn new(wave_function: &'a mut T, metrop: V) -> Self {
        let nelec = wave_function.num_electrons();
        Self{
            wave_function,
            config: Array2::<f64>::random((nelec, 3), Range::new(-1., 1.)),
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
    means: Vec<f64>
}

impl<S> Runner<S>
where S: MonteCarloSampler
{
    pub fn new(sampler: S) -> Self {
        let mut means = Vec::<f64>::new();
        means.resize(sampler.num_observables(), 0.0);
        Self{sampler, means}
    }

    pub fn run(&mut self, iters: usize) {
        let nelec = self.sampler.num_electrons();
        let mut count = 0_usize;
        for iter in 0..iters {
            for e in 0..nelec {
                count += 1;
                self.sampler.move_state(e);
                let samples = self.sampler.sample().expect("Failed to sample observables");
                self.means.iter_mut().zip(samples.iter()).for_each(|(x, y)| {
                    *x = (count as f64 * *x + y)/(count as f64 + 1.0)
                });
            }
        }
    }

    pub fn means(&self) -> &Vec<f64> {
        &self.means
    }

}