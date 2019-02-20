// Standard imports
use std::collections::HashMap;
// First party imports
use crate::traits::*;
use crate::block::Block;

/// Struct for running Monte Carlo integration
/// Generic over Samplers
pub struct Runner<S: MonteCarloSampler> {
    sampler: S,
    means: HashMap<String, f64>,
    variances: HashMap<String, f64>,
    square_mean_diff: HashMap<String, f64>,
}

impl<S> Runner<S>
    where S: MonteCarloSampler
{
    pub fn new(sampler: S) -> Self {
        // initialize means at a sample of the current configuration
        let means = sampler.sample().expect("Failed to perform initial sampling");
        let variances = means.keys().map(|key| (key.clone(), 0.0)).collect();
        let square_mean_diff = means.keys().map(|key| (key.clone(), 0.0)).collect();
        Self{sampler, means, variances, square_mean_diff}
    }

    pub fn run(&mut self, steps: usize, block_size: usize) {
        // needed for pretty printing output
        let max_strlen = self.means.keys().map(|key| key.len()).max().unwrap();
        assert!(steps >= 2*block_size);
        let blocks = steps / block_size;
        for block_nr in 0..blocks {
            let mut block = Block::new(block_size, &self.sampler.observable_names());

            for b in 0..block_size {
                self.sampler.move_state();
                // Discard first block for equilibration
                if block_nr > 0 {
                    let samples = self.sampler.sample()
                        .expect("Failed to sample observables");
                    block.set_value(b, &samples);
                }
            }
            if block_nr > 0 {
                // compute block mean
                let block_mean = block.mean();
                self.update_means_and_variances(block_nr, &block_mean);
                // log output
                // TODO: find better way to log output
                for key in block_mean.keys() {
                    let mean = self.means.get(key).unwrap();
                    let var = self.variances.get(key).unwrap();

                    let padding = max_strlen - key.len() + if *mean < 0.0 { 3 } else { 4 };
                    println!("{}:{:>width$} {:.*} +/- {:.*}", key, "", 8, mean, 8, var, width=padding);
                }
            }

        }
    }

    pub fn means(&self) -> HashMap<&str, f64> {
        self.means.iter().map(|(key, value)| (key.as_str(), *value)).collect()
    }

    pub fn variances(&self) -> HashMap<&str, f64> {
        self.variances.iter().map(|(key, value)| (key.as_str(), *value)).collect()
    }

    fn update_means_and_variances(&mut self, idx: usize, block_mean: &HashMap<String, f64>) {
        let old_mean = self.means.clone();
        for (name, current_mean) in self.means.iter_mut() {
            let bm = block_mean.get(name).unwrap();
            let om = old_mean.get(name).unwrap();
            let smd = self.square_mean_diff.get_mut(name).unwrap();
            let var = self.variances.get_mut(name).unwrap();

            // update running mean
            *current_mean += (bm - *current_mean)/idx as f64;
            // update square mean sifference
            *smd += (bm - om)*(bm - *current_mean);
            // update running variance
            *var = *smd/idx as f64;
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use basis;
    use wavefunction::{Orbital, SingleDeterminant};
    use operator::{
        ElectronicPotential,
        IonicPotential, 
        KineticEnergy, 
        ElectronicHamiltonian
    };
    use metropolis::MetropolisBox;
    use crate::samplers::Sampler;

    #[test]
    fn test_hydrogen_atom_single_det_metrop_box() {
        // Tests the monte carlo result for a single hydrogen atom
        const ENERGY_EXACT: f64 = -0.5;
        let basis_set: Vec<Box<basis::Func>> = vec![
            Box::new(|x| basis::hydrogen_1s(x, 1.0))
        ];
        let orbital = Orbital::new(array![1.0], &basis_set);
        let wave_func = SingleDeterminant::new(vec![orbital]);
        let local_e = ElectronicHamiltonian::new(
            KineticEnergy::new(),
            IonicPotential::new(array![[0., 0., 0.]], array![1]),
            ElectronicPotential::new()
        );
        let metropolis = MetropolisBox::<SmallRng>::new(1.0);
        let mut sampler = Sampler::new(wave_func, metropolis);
        sampler.add_observable("Local Energy", local_e);

        let mut runner = Runner::new(sampler);
        runner.run(100, 1);

        let result = *runner.means().get("Local Energy").unwrap();

        assert!((result - ENERGY_EXACT).abs() < 1e-15);
    }

}
