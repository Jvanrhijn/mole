// Standard imports
use std::vec::Vec;
//Third party imports
use ndarray::{Ix2, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
use rand::Rng;
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
pub struct Sampler<T, V>
where T: Function<f64, D=Ix2> + Differentiate + Cache<Array2<f64>>,
      V: Metropolis<T>
{
    wave_function: T,
    config: Array2<f64>,
    metropolis: V,
    observables: Vec<Box<Operator<T>>>,
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
            observables: Vec::<Box<Operator<T>>>::new(),
        }
    }

    pub fn add_observable<O>(&mut self, operator: O)
    where O: 'static + Operator<T>
    {
        self.observables.push(Box::new(operator));
    }
}

impl<T, V> MonteCarloSampler for Sampler<T, V>
where T: Function<f64, D=Ix2> + Differentiate + WaveFunction + Cache<Array2<f64>, U=usize>,
      V: Metropolis<T>
{
    fn sample(&self) -> Result<Vec<f64>, Error> {
        Ok(self.observables.iter().map(|x| x.act_on(&self.wave_function, &self.config)
            .expect("Failed to act on wave function with operator")).collect())
    }

    fn move_state(&mut self) {
        for e in 0..self.wave_function.num_electrons() {
            if let Some(config) = self.metropolis.move_state(&mut self.wave_function, &self.config, e) {
                self.config = config;
                self.wave_function.push_update();
            } else {
                self.wave_function.flush_update();
            }
        }
        self.wave_function.refresh(&self.config);
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
    variances: Vec<f64>,
    square_mean_diff: Vec<f64>,
    mean_sq: Vec<f64>
}

impl<S> Runner<S>
where S: MonteCarloSampler
{
    pub fn new(sampler: S) -> Self {
        // initialize means at a sample of the current configuration
        let means = sampler.sample().expect("Failed to sample observables");
        let variances = vec![0.0; means.len()];
        let square_mean_diff = vec![0.0; means.len()];
        let mean_sq: Vec<f64> = means.iter().map(|x| x.powi(2)).collect();
        Self{sampler, means, variances, square_mean_diff, mean_sq}
    }

    pub fn run(&mut self, steps: usize, block_size: usize) {
        assert!(steps >= 2*block_size);
        let blocks = steps / block_size;
        for block_nr in 0..blocks {
            let mut block = Block::new(block_size, self.sampler.num_observables());
            for b in 0..block_size {
                self.sampler.move_state();
                // Discard first block for equilibration
                if block_nr > 0 {
                    let samples = self.sampler.sample()
                        .expect("Failed to sample observables");
                    println!("{}", samples[0]);
                    block.set_value(b, samples);
                }
            }
            // TODO: write abstraction over statistics
            if block_nr > 0 {
                let block_mean = block.mean();
                // running mean algorithm
                izip!(self.means.iter_mut(), block_mean.iter())
                    .for_each(|(m, x)| *m += x);
                izip!(self.mean_sq.iter_mut(), block_mean.iter())
                    .for_each(|(msq, m)| *msq += m.powi(2));
            }
        }
        izip!(self.means.iter_mut(), self.mean_sq.iter_mut()).for_each(|(m, msq)| {
            *m /= blocks as f64;
            *msq /= blocks as f64;
        });
        izip!(self.variances.iter_mut(), self.means.iter(), self.mean_sq.iter())
            .for_each(|(v, m, msq)| *v = msq - m.powi(2));
    }

    pub fn means(&self) -> &Vec<f64> {
        &self.means
    }

    pub fn variances(&self) -> &Vec<f64> {
        &self.variances
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
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
        let wave_func = wf::SingleDeterminant::new(vec![orbital]);
        let local_e = LocalEnergy::new(
            ElectronicHamiltonian::new(
                KineticEnergy::new(),
                IonicPotential::new(array![[0., 0., 0.]], array![1]),
                ElectronicPotential::new()
            )
        );
        let metropolis = MetropolisBox::<SmallRng>::new(1.0);
        let mut sampler = Sampler::new(wave_func, metropolis);
        sampler.add_observable(local_e);

        let mut runner = Runner::new(sampler);
        runner.run(100, 1);

        let local_e_result = runner.means()[0];

        assert!((local_e_result - ENERGY_EXACT).abs() < 1e-15);
    }

}
