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
