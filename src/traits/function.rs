// Third pary imports
use ndarray::{Array};
// First party imports
use error::{Error};

pub trait Function<T> {

    type D;

    fn value(&self, cfg: &Array<T, Self::D>) -> Result<T, Error>;
}
