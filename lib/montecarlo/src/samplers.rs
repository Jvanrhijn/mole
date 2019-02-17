// Standard imports
use std::collections::HashMap;
//Third party imports
use ndarray::{Ix2, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
use rand::Rng;
// First party imports
use crate::traits::*;
use wavefunction::{Differentiate, Function, WaveFunction, Cache, Error};
use metropolis::Metropolis;
use operator::Operator;

/// Simple Monte Carlo sampler
/// Performs Metropolis step and keeps list of observables to sample
pub struct Sampler<T, V>
    where T: Function<f64, D=Ix2> + Differentiate + Cache<Array2<f64>>,
          V: Metropolis<T>
{
    wave_function: T,
    config: Array2<f64>,
    metropolis: V,
    observables: HashMap<String, Box<Operator<T>>>,
    acceptance: f64
}

impl<T, V> Sampler<T, V>
    where T: Function<f64, D=Ix2> + Differentiate + WaveFunction + Cache<Array2<f64>, V=(f64, f64)>,
          V: Metropolis<T>,
          <V as Metropolis<T>>::R: Rng
{
    pub fn new(mut wave_function: T, mut metrop: V) -> Self {
        let nelec = wave_function.num_electrons();
        let cfg = Array2::<f64>::random_using((nelec, 3), Range::new(-1., 1.), metrop.rng_mut());
        wave_function.refresh(&cfg);
        Self{
            wave_function,
            config: cfg,
            metropolis: metrop,
            observables: HashMap::<String, Box<Operator<T>>>::new(),
            acceptance: 0.0
        }
    }

    pub fn add_observable<O>(&mut self, name: &str, operator: O)
        where O: 'static + Operator<T>
    {
        self.observables.insert(name.to_string(), Box::new(operator));
    }
}

impl<T, V> MonteCarloSampler for Sampler<T, V>
    where T: Function<f64, D=Ix2> + Differentiate + WaveFunction + Cache<Array2<f64>, U=usize>,
          V: Metropolis<T>
{
    fn sample(&self) -> Result<HashMap<String, f64>, Error> {
        Ok(self.observables.iter()
            .map(|(name, operator)| {
                (name.clone(), operator.act_on(&self.wave_function, &self.config)
                    .expect("Failed to act on wave function with operator"))
            }
        ).collect())
    }

    fn move_state(&mut self) {
        for e in 0..self.wave_function.num_electrons() {
            if let Some(config) = self.metropolis.move_state(&mut self.wave_function, &self.config, e) {
                self.config = config;
                self.wave_function.push_update();
                self.acceptance += 1.0/self.wave_function.num_electrons() as f64;
            } else {
                self.wave_function.flush_update();
            }
        }
        self.wave_function.refresh(&self.config);
    }

    fn num_observables(&self) -> usize {
        self.observables.len()
    }

    fn acceptance(&self) -> f64 {
        self.acceptance
    }

}
