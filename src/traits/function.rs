// Standard imports
use std::fmt::Debug;
// Third pary imports
use ndarray::{Array};
// First party imports
use error::{FuncError, Error};

pub trait Function<T> {

    type D;

    fn value(&self, cfg: &Array<T, Self::D>) -> Result<T, Error>;
}
