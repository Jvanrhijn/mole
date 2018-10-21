// Standard imports
use std::vec::Vec;
//Third party imports
use ndarray::{Ix2, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
// First party imports
use traits::mcsamplers::*;
use traits::differentiate::Differentiate;
use traits::function::Function;
use traits::metropolis::Metropolis;
use traits::operator::Operator;
use traits::wavefunction::WaveFunction;
use error::Error;
use block::Block;


/// Simple Monte Carlo sampler
/// Performs Metropolis step and keeps list of observables to sample
pub struct Sampler<'a, T, V>
where T: Function<f64, D=Ix2> + Differentiate,
      V: Metropolis<T>,
{
    wave_function: &'a mut T,
    config: Array2<f64>,
    metropolis: V,
    observables: Vec<Box<Operator<T>>>
}

impl<'a, T, V> Sampler<'a, T, V>
where T: Function<f64, D=Ix2> + Differentiate + WaveFunction,
      V: Metropolis<T>,
{
    pub fn new(wave_function: &'a mut T, mut metrop: V) -> Self {
        let nelec = wave_function.num_electrons();
        let cfg = Array2::<f64>::random((nelec, 3), Range::new(-1., 1.));
        metrop.set_wave_function_value(wave_function.value(&cfg)
            .expect("Failed to evaluate wave function"));
        Self{
            wave_function,
            config: cfg,
            metropolis: metrop,
            observables: Vec::<Box<Operator<T>>>::new(),
        }
    }

    pub fn add_observable<O>(&mut self, operator: O)
    where O: 'static + Operator<T>
    {
        self.observables.push(Box::new(operator));
    }
}

impl<'a, T, V> MonteCarloSampler for Sampler<'a, T, V>
where T: Function<f64, D=Ix2> + Differentiate + WaveFunction,
      V: Metropolis<T>,
{
    fn sample(&self) -> Result<Vec<f64>, Error> {
        Ok(self.observables.iter().map(|x| x.act_on(self.wave_function, &self.config)
            .expect("Failed to act on wave function with operator")).collect())
    }

    fn move_state(&mut self) {
        for e in 0..self.wave_function.num_electrons() {
            if let Some(config) = self.metropolis.move_state(self.wave_function, &self.config, e) {
                self.config = config;
            }
        }
    }

    fn num_observables(&self) -> usize {
        self.observables.len()
    }

}

/// Struct for running Monte Carlo integration
/// Generic over Samplers
pub struct Runner<S: MonteCarloSampler> {
    sampler: S,
    means: Vec<f64>,
}

impl<S> Runner<S>
where S: MonteCarloSampler
{
    pub fn new(sampler: S) -> Self {
        let mut means = Vec::<f64>::new();
        means.resize(sampler.num_observables(), 0.0);
        Self{sampler, means}
    }

    pub fn run(&mut self, blocks: usize, block_size: usize) {
        for block_nr in 0..blocks {
            let mut block = Block::new(block_size, self.sampler.num_observables());
            for b in 0..block_size {
                self.sampler.move_state();
                // Discard first block for equilibration
                if block_nr > 0 {
                    let samples = self.sampler.sample()
                        .expect("Failed to sample observables");
                    block.set_value(b, samples);
                }
            }
            self.means.iter_mut().zip(block.mean().iter())
                .for_each(|(m, x)| *m = (x + block_nr as f64 * *m)/(block_nr + 1) as f64);
            if block_nr > 0 {
                println!("Local E = {:.*}", 5, self.means[0]);
            }
        }
    }

    pub fn means(&self) -> &Vec<f64> {
        &self.means
    }

}