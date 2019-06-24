// Standard imports
use std::collections::HashMap;
// First party imports
use crate::block::Block;
use crate::traits::*;
use operator::OperatorValue;

/// Struct for running Monte Carlo integration
/// Generic over Samplers
pub struct Runner<S: MonteCarloSampler> {
    sampler: S,
    means: HashMap<String, f64>,
    errors: HashMap<String, f64>,
    square_mean_diff: HashMap<String, f64>,
}

impl<S> Runner<S>
where
    S: MonteCarloSampler,
{
    pub fn new(sampler: S) -> Self {
        // initialize means at a sample of the current configuration
        let means: HashMap<String, f64> = sampler
            .sample()
            .expect("Failed to perform initial sampling")
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    match v {
                        OperatorValue::Scalar(v) => *v,
                        _ => unimplemented!(),
                    },
                )
            })
            .collect();
        let errors = means.keys().map(|key| (key.clone(), 0.0)).collect();
        let square_mean_diff = means.keys().map(|key| (key.clone(), 0.0)).collect();
        Self {
            sampler,
            means,
            errors,
            square_mean_diff,
        }
    }

    pub fn run(&mut self, steps: usize, block_size: usize) {
        // needed for pretty printing output
        let max_strlen = self.means.keys().map(|key| key.len()).max().unwrap();
        assert!(steps >= 2 * block_size);
        let blocks = steps / block_size;
        for block_nr in 0..blocks {
            let mut block = Block::new(block_size, &self.sampler.observable_names());

            for b in 0..block_size {
                self.sampler.move_state();
                // Discard first block for equilibration
                if block_nr > 0 {
                    let samples: HashMap<String, f64> = self
                        .sampler
                        .sample()
                        .expect("Failed to sample observables")
                        .iter()
                        .map(|(k, v)| {
                            (
                                k.clone(),
                                match v {
                                    OperatorValue::Scalar(value) => *value,
                                    _ => unimplemented!(),
                                },
                            )
                        })
                        .collect();
                    block.set_value(b, &samples);
                }
            }
            if block_nr > 0 {
                // compute block mean
                let block_mean = block.mean();
                self.update_means_and_variances(block_nr, &block_mean);
                // log output
                self.log_data(&block_mean, max_strlen);
                //println!("{}", self.sampler.acceptance() / (block_nr * block_size) as f64);
            }
        }
    }

    pub fn means(&self) -> HashMap<&str, f64> {
        self.means
            .iter()
            .map(|(key, value)| (key.as_str(), *value))
            .collect()
    }

    pub fn errors(&self) -> HashMap<&str, f64> {
        self.errors
            .iter()
            .map(|(key, value)| (key.as_str(), *value))
            .collect()
    }

    fn update_means_and_variances(&mut self, idx: usize, block_mean: &HashMap<String, f64>) {
        let old_mean = self.means.clone();
        for (name, current_mean) in self.means.iter_mut() {
            let bm = block_mean.get(name).unwrap();
            let om = old_mean.get(name).unwrap();
            let smd = self.square_mean_diff.get_mut(name).unwrap();
            let error = self.errors.get_mut(name).unwrap();

            // update running mean
            *current_mean += (bm - *current_mean) / idx as f64;
            // update square mean sifference
            *smd += (bm - om) * (bm - *current_mean);
            // update running variance
            *error = (*smd / idx as f64).sqrt() / (idx as f64).sqrt();
        }
    }

    fn log_data(&self, block_mean: &HashMap<String, f64>, max_strlen: usize) {
        // TODO: find better way to log output
        for key in self.means.keys() {
            let mean = self.means.get(key).unwrap();
            let error = self.errors.get(key).unwrap();

            let padding = max_strlen - key.len() + if *mean < 0.0 { 3 } else { 4 };
            let padding2 = if *mean < 0.0 { 3 } else { 4 };
            println!(
                "{}:{:>width$} {:.*} {:>width2$}{:.*} error {:.*}",
                key,
                "",
                16,
                block_mean.get(key).unwrap(),
                "",
                16,
                mean,
                16,
                error,
                width = padding,
                width2 = padding2
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samplers::Sampler;
    use basis::{self, Hydrogen1sBasis};
    use metropolis::MetropolisBox;
    use operator::{ElectronicHamiltonian, ElectronicPotential, IonicPotential, KineticEnergy};
    use rand::rngs::StdRng;
    use wavefunction::{Orbital, SingleDeterminant};

    #[test]
    fn test_hydrogen_atom_single_det_metrop_box() {
        // Tests the monte carlo result for a single hydrogen atom
        const ENERGY_EXACT: f64 = -0.5;
        let basis_set = Hydrogen1sBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0]);
        let orbital = Orbital::new(array![[1.0]], basis_set);
        let wave_func = SingleDeterminant::new(vec![orbital]);

        let local_e = ElectronicHamiltonian::new(
            KineticEnergy::new(),
            IonicPotential::new(array![[0., 0., 0.]], array![1]),
            ElectronicPotential::new(),
        );

        let metropolis = MetropolisBox::<StdRng>::new(1.0);
        let mut sampler = Sampler::new(wave_func, metropolis);
        sampler.add_observable("Local Energy", local_e);

        let mut runner = Runner::new(sampler);
        runner.run(100, 1);

        let result = *runner.means().get("Local Energy").unwrap();

        assert!((result - ENERGY_EXACT).abs() < 1e-15);
    }

}
