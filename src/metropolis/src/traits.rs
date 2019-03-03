use ndarray::{Array2, Ix2};
use wavefunction::{Differentiate, Function};

/// Interface for implementing Metropolis algorithms that generate
/// Markov chains of configurations.
pub trait Metropolis<T: Differentiate + Function<f64, D = Ix2>> {
    /// Rng type to use
    type R;
    /// Propose a move to a new configuration.
    fn propose_move(&mut self, wf: &mut T, cfg: &Array2<f64>, idx: usize) -> Array2<f64>;
    /// Test whether a proposed configuration will be accepted.
    fn accept_move(&mut self, wf: &mut T, cfg: &Array2<f64>, cfg_prop: &Array2<f64>) -> bool;
    /// Return an Option containing the new configuration if it was accepted,
    /// and None otherwise.
    fn move_state(&mut self, wf: &mut T, cfg: &Array2<f64>, idx: usize) -> Option<Array2<f64>>;
    // Get a mut ref to the internal rng
    fn rng_mut(&mut self) -> &mut Self::R;
}
