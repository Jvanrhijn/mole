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
use block::Block;


pub struct Sampler<'a, T, V>
where T: Function<f64, D=Ix2> + WaveFunction,
      V: Metropolis<T>,
{
    wave_function: &'a mut T,
    config: Array2<f64>,
    metropolis: V,
    observables: Vec<Box<Operator<T>>>
}

impl<'a, T, V> Sampler<'a, T, V>
where T: Function<f64, D=Ix2> + WaveFunction,
      V: Metropolis<T>,
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
where T: Function<f64, D=Ix2> + WaveFunction,
      V: Metropolis<T>,
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
}

impl<S> Runner<S>
where S: MonteCarloSampler
{
    pub fn new(sampler: S) -> Self {
        let means = Vec::<f64>::new();
        Self{sampler, means}
    }

    pub fn run(&mut self, blocks: usize, block_size: usize) {
        let nelec = self.sampler.num_electrons();
        for b in 0..blocks {
            let mut block_count = 0;
            let mut block = Block::new(nelec*block_size);
            for _ in 0..block_size {
                for e in 0..nelec {
                    self.sampler.move_state(e);
                    let samples = self.sampler.sample()
                        .expect("Failed to sample observables");
                    if b > 0 { // discard first block for equilibration
                        * block.value_mut(block_count) = samples[0];
                    block_count += 1;
                    }
                }
            }
            println!("Block average = {}", block.mean());
        }
    }

    pub fn means(&self) -> &Vec<f64> {
        &self.means
    }

}