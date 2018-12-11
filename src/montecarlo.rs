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
use traits::cache::Cache;
use error::Error;
use block::Block;


/// Simple Monte Carlo sampler
/// Performs Metropolis step and keeps list of observables to sample
pub struct Sampler<'a, T, V>
where T: Function<f64, D=Ix2> + Differentiate + Cache<Array2<f64>>,
      V: Metropolis<T>,
{
    wave_function: &'a mut T,
    config: Array2<f64>,
    metropolis: V,
    observables: Vec<Box<Operator<T>>>
}

impl<'a, T, V> Sampler<'a, T, V>
where T: Function<f64, D=Ix2> + Differentiate + WaveFunction + Cache<Array2<f64>, V=(f64, f64)>,
      V: Metropolis<T>,
{
    pub fn new(wave_function: &'a mut T, mut metrop: V) -> Self {
        let nelec = wave_function.num_electrons();
        let cfg = Array2::<f64>::random((nelec, 3), Range::new(-1., 1.));
        wave_function.refresh(&cfg);
        metrop.set_wave_function_value(wave_function.current_value().0);
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
where T: Function<f64, D=Ix2> + Differentiate + WaveFunction + Cache<Array2<f64>, U=usize>,
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
                self.wave_function.push_update();
            } else {
                self.wave_function.flush_update();
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
        // initialize means at a sample of the current configuration
        let means = sampler.sample().expect("Failed to sample observables");
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
            // TODO: write abstraction over statistics
            if block_nr > 0 {
                self.means.iter_mut().zip(block.mean().iter())
                    .for_each(|(m, x)| *m = (x + block_nr as f64 * *m)/(block_nr + 1) as f64);
            }
        }
    }

    pub fn means(&self) -> &Vec<f64> {
        &self.means
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use math::basis;
    use ndarray::Array1;
    use orbitals::Orbital;
    use wf;
    use operators::{LocalEnergy, ElectronicPotential, IonicPotential, KineticEnergy, ElectronicHamiltonian};
    use metrop::MetropolisBox;

    #[test]
    fn test_hydrogen_atom_single_det_metrop_box() {
        // Tests the monte carlo result for a single hydrogen atom
        const ENERGY_EXACT: f64 = -0.5;
        let basis_set: Vec<Box<Fn(&Array1<f64>) -> (f64, f64)>> = vec![
            Box::new(basis::hydrogen_1s)
        ];
        let orbital = Orbital::new(array![1.0], &basis_set);
        let mut wave_func = wf::SingleDeterminant::new(vec![orbital]);
        let local_e = LocalEnergy::new(
            ElectronicHamiltonian::new(
                KineticEnergy::new(),
                IonicPotential::new(array![[0., 0., 0.]], array![1]),
                ElectronicPotential::new()
            )
        );
        let metropolis = MetropolisBox::new(1.0);
        let mut sampler = Sampler::new(&mut wave_func, metropolis);
        sampler.add_observable(local_e);

        let mut runner = Runner::new(sampler);
        runner.run(10, 1);

        let local_e_result = runner.means()[0];
        println!("{} {}", local_e_result, ENERGY_EXACT);

        assert!((local_e_result - ENERGY_EXACT).abs() < 1e-15);
    }

}
