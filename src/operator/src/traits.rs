// Third party imports
use ndarray::Array2;
use wavefunction::Error;

/// Interface for creating quantum operators that act on Function types.
pub trait Operator<T> {
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<f64, Error>;
}
