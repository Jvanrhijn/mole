use std::vec::Vec;
use error::Error;
use traits::wavefunction::WaveFunction;

pub trait MonteCarloSampler {
    fn sample(&self) -> Result<Vec<f64>, Error>;

    fn move_state(&mut self, elec: usize);

    fn num_electrons(&self) -> usize;

    fn num_observables(&self) -> usize;
}
