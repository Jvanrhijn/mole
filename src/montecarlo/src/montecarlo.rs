// Standard imports
use std::collections::HashMap;
// First party imports
use crate::block::Block;
use crate::traits::*;
use operator::OperatorValue;

/// Struct for running Monte Carlo integration
/// Generic over Samplers
pub struct Runner<S: MonteCarloSampler, L: Log> {
    sampler: S,
    logger: L,
}

impl<S, L> Runner<S, L>
where
    S: MonteCarloSampler,
    L: Log
{
    pub fn new(sampler: S, logger: L) -> Self {
        Self {
            sampler,
            logger,
        }
    }

    pub fn run(&mut self, steps: usize, block_size: usize) {
        assert!(steps >= 2 * block_size);
        let blocks = steps / block_size;
        for block_nr in 0..blocks {
            for b in 0..block_size {
                self.sampler.move_state();
                // Discard first block for equilibration
                if block_nr > 0 {
                    self.sampler.sample();
                    println!("{}", self.logger.log(self.sampler.data()));
                }
            }
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
