// Standard imports
use std::collections::HashMap;
// First party imports
use crate::traits::*;
use errors::Error;
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
    L: Log,
{
    pub fn new(sampler: S, logger: L) -> Self {
        Self { sampler, logger }
    }

    pub fn run(
        mut self,
        steps: usize,
        block_size: usize,
    ) -> Result<MonteCarloResult<S::WaveFunc>, Error> {
        assert!(steps >= 2 * block_size);
        let blocks = steps / block_size;
        let mut output = String::new();
        for block_nr in 0..blocks {
            for _ in 0..block_size {
                self.sampler.move_state()?;
                // Discard first block for equilibration
                if block_nr > 0 {
                    self.sampler.sample()?;
                    output = self.logger.log(self.sampler.data());
                }
            }
            if !output.is_empty() {
                println!("{}", output);
            }
        }
        Ok(self.sampler.consume_result())
    }

    pub fn data(&self) -> &HashMap<String, Vec<OperatorValue>> {
        self.sampler.data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samplers::Sampler;
    use crate::traits::Log;
    use basis::{self, Hydrogen1sBasis};
    use metropolis::MetropolisBox;
    use operator::{
        ElectronicHamiltonian, ElectronicPotential, IonicPotential, KineticEnergy, LocalOperator,
    };
    use rand::rngs::StdRng;
    use wavefunction::{Orbital, SingleDeterminant};

    struct MockLogger;
    impl Log for MockLogger {
        fn log(&mut self, _data: &HashMap<String, Vec<OperatorValue>>) -> String {
            String::new()
        }
    }

    #[test]
    fn test_hydrogen_atom_single_det_metrop_box() {
        // Tests the monte carlo result for a single hydrogen atom
        const ENERGY_EXACT: f64 = -0.5;
        let basis_set = Hydrogen1sBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0]);
        let orbital = Orbital::new(array![[1.0]], basis_set);
        let wave_func = SingleDeterminant::new(vec![orbital]).unwrap();

        let local_e = ElectronicHamiltonian::new(
            KineticEnergy::new(),
            IonicPotential::new(array![[0., 0., 0.]], array![1]),
            ElectronicPotential::new(),
        );

        let metropolis = MetropolisBox::<StdRng>::new(1.0);
        let mut obs = HashMap::new();
        obs.insert(
            "Local Energy".to_string(),
            Box::new(local_e) as Box<dyn LocalOperator<_>>,
        );
        let sampler = Sampler::new(wave_func, metropolis, &obs).unwrap();

        let sampler = Runner::new(sampler, MockLogger).run(100, 1).unwrap();

        let result = sampler
            .data
            .get("Local Energy")
            .unwrap()
            .clone()
            .into_iter()
            .sum::<OperatorValue>()
            / OperatorValue::Scalar(99.0);

        assert!((result.get_scalar().unwrap() - ENERGY_EXACT).abs() < 1e-15);
    }
}
