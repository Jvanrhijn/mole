use std::vec::Vec;
use error::Error;

/// Interface for sampling observables from Monte Carlo integration.
pub trait MonteCarloSampler {
    /// Sample observables from the current configuration.
    fn sample(&self) -> Result<Vec<f64>, Error>;

    /// Move the current state to a new configuration.
    fn move_state(&mut self, elec: usize);

    fn num_electrons(&self) -> usize;

    fn num_observables(&self) -> usize;
}
