use errors::Error;
use operator::OperatorValue;
use std::collections::HashMap;

pub struct MonteCarloResult<T> {
    pub wave_function: T,
    pub acceptance: f64,
    pub data: HashMap<String, Vec<OperatorValue>>,
}

/// Interface for sampling observables from Monte Carlo integration.
pub trait MonteCarloSampler {
    type WaveFunc;

    /// Sample observables from the current configuration.
    fn sample(&mut self) -> Result<(), Error>;

    /// Get the data sampled thus far
    fn data(&self) -> &HashMap<String, Vec<OperatorValue>>;

    /// Move the current state to a new configuration.
    fn move_state(&mut self) -> Result<(), Error>;

    fn num_observables(&self) -> usize;

    fn acceptance(&self) -> f64;

    fn observable_names(&self) -> Vec<&String>;

    fn consume_result(self) -> MonteCarloResult<Self::WaveFunc>;

    fn wave_function(&self) -> &Self::WaveFunc;

    fn wave_function_mut(&mut self) -> &mut Self::WaveFunc;

    fn reseed_rng(&mut self, seed: [u8; 32]);

    fn generate_seed(&mut self) -> [u8; 32];
}

/// Trait for creating a logging configuration. Implement
/// this trait in order to get custom output for a Monte Carlo
/// run.
pub trait Log {
    /// Use logging data to create pretty logging output string.
    fn log(&mut self, data: &HashMap<String, Vec<OperatorValue>>) -> String;
}
