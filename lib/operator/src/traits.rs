// Third party imports
use ndarray::{Array2};
/// First party imports
use wavefunction::Function;
use wavefunction::Error;

/// Interface for creating quantum operators that act on Function types.
pub trait Operator<T>
    where T: Function<f64> + ?Sized
{
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<f64, Error>;
}
