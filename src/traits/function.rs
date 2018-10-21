// Third pary imports
use ndarray::{Array};
// First party imports
use error::{Error};

/// Interface for dealing with functions f: F^n -> F, where F is any field.
pub trait Function<T> {

    type D;

    fn value(&self, cfg: &Array<T, Self::D>) -> Result<T, Error>;
}
