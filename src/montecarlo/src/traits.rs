use operator::OperatorValue;
use std::collections::HashMap;
use wavefunction::Error;

/// Interface for sampling observables from Monte Carlo integration.
pub trait MonteCarloSampler {
    /// Sample observables from the current configuration.
    fn sample(&mut self);

    /// Get the data sampled thus far
    fn data(&self) -> &HashMap<String, Vec<OperatorValue>>;

    /// Move the current state to a new configuration.
    fn move_state(&mut self);

    fn num_observables(&self) -> usize;

    fn acceptance(&self) -> f64;

    fn observable_names(&self) -> Vec<&String>;
}

/// Trait for creating a logging configuration. Implement
/// this trait in order to get custom output for a Monte Carlo
/// run.
pub trait Log {
    /// Use logging data to create pretty logging output string.
    fn log(&self, data: &HashMap<String, Vec<OperatorValue>>) -> String;
}
