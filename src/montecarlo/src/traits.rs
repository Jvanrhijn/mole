use operator::OperatorValue;
use std::collections::HashMap;
use wavefunction::Error;

/// Interface for sampling observables from Monte Carlo integration.
pub trait MonteCarloSampler {
    /// Sample observables from the current configuration.
    fn sample(&self) -> Result<HashMap<String, OperatorValue>, Error>;

    /// Move the current state to a new configuration.
    fn move_state(&mut self);

    fn num_observables(&self) -> usize;

    fn acceptance(&self) -> f64;

    fn observable_names(&self) -> Vec<&String>;
}
