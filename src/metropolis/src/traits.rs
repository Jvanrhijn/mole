use errors::Error;
use ndarray::{Array2, Ix2};
use rand::Rng;
use wavefunction_traits::{Differentiate, Function};

type Result<T> = std::result::Result<T, Error>;

/// Interface for implementing Metropolis algorithms that generate
/// Markov chains of configurations.
pub trait Metropolis<T: Differentiate + Function<f64, D = Ix2> + Clone> {
    /// Rng type to use
    type R: Rng;
    /// Propose a move to a new configuration.
    fn propose_move(&mut self, wf: &mut T, cfg: &Array2<f64>, idx: usize) -> Result<Array2<f64>>;
    /// Test whether a proposed configuration will be accepted.
    fn accept_move(
        &mut self,
        wf: &mut T,
        cfg: &Array2<f64>,
        cfg_prop: &Array2<f64>,
    ) -> Result<bool>;
    /// Return an Option containing the new configuration if it was accepted,
    /// and None otherwise.
    fn move_state(
        &mut self,
        wf: &mut T,
        cfg: &Array2<f64>,
        idx: usize,
    ) -> Result<Option<Array2<f64>>>;
    // Get a mut ref to the internal rng
    fn rng_mut(&mut self) -> &mut Self::R;
    /// Reseed the internal rng
    fn reseed_rng(&mut self, s: [u8; 32]);
    // Generate new seed
    fn generate_seed(&mut self) -> [u8; 32] {
        self.rng_mut().gen::<[u8; 32]>()
    }
}
