// Standard imports
use std::vec::Vec;
use ndarray::Array1;
// First party imports
use crate::traits::*;
use crate::block::Block;

/// Struct for running Monte Carlo integration
/// Generic over Samplers
pub struct Runner<S: MonteCarloSampler> {
    sampler: S,
    means: Vec<f64>,
    variances: Vec<f64>,
    square_mean_diff: Vec<f64>,
}

impl<S> Runner<S>
    where S: MonteCarloSampler
{
    pub fn new(sampler: S) -> Self {
        // initialize means at a sample of the current configuration
        let means = sampler.sample().expect("Failed to sample observables");
        let variances = vec![0.0; means.len()];
        let square_mean_diff = vec![0.0; means.len()];
        Self{sampler, means, variances, square_mean_diff}
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
                    block.set_value(b, samples);
                }
            }
            if block_nr > 0 {
                let block_mean = block.mean();
                self.update_means_and_variances(block_nr, &block_mean);
                let acceptance = self.sampler.acceptance()/(block_nr*block_size) as f64;
                println!("{:.*}    {:.*} +/- {:.*}    acc {:.*}",
                         8, block_mean[0], 8, self.means[0], 8, self.variances[0].sqrt(), 8, acceptance);
            }
        }
        println!("{}", self.sampler.acceptance()/(blocks*block_size) as f64);
    }

    pub fn means(&self) -> &Vec<f64> {
        &self.means
    }

    pub fn variances(&self) -> &Vec<f64> {
        &self.variances
    }

    fn update_means_and_variances(&mut self, idx: usize, block_mean: &Array1<f64>) {
        // running mean algorithm
        let old_mean = self.means.clone();
        izip!(self.means.iter_mut(), block_mean.iter())
            .for_each(|(m, x)| *m += (x - *m)/idx as f64);
        izip!(self.square_mean_diff.iter_mut(), block_mean.iter(), old_mean.iter(), self.means.iter())
            .for_each(|(m2, x, xbarold, xbar)| *m2 += (x - xbarold)*(x - xbar));
        self.variances = self.square_mean_diff.iter().map(|m2| m2/idx as f64).collect();
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use basis;
    use ndarray::Array1;
    use wavefunction::{Orbital, SingleDeterminant};
    use operator::{LocalEnergy, ElectronicPotential, IonicPotential, KineticEnergy, ElectronicHamiltonian};
    use metropolis::MetropolisBox;
    use crate::samplers::Sampler;

    #[test]
    fn test_hydrogen_atom_single_det_metrop_box() {
        // Tests the monte carlo result for a single hydrogen atom
        const ENERGY_EXACT: f64 = -0.5;
        let basis_set: Vec<Box<Fn(&Array1<f64>) -> (f64, f64)>> = vec![
            Box::new(basis::hydrogen_1s)
        ];
        let orbital = Orbital::new(array![1.0], &basis_set);
        let wave_func = SingleDeterminant::new(vec![orbital]);
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
