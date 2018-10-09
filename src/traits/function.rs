// Standard imports
use std::fmt::Debug;
// Third pary imports
use ndarray::{Array};

pub trait Function<T> {

    type E: Debug;
    type D;

    fn value(&self, cfg: &Array<T, Self::D>) -> Result<T, Self::E>;
}
