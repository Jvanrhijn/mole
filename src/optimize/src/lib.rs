#[macro_use]
extern crate itertools;
#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;

pub mod optimizers;
pub mod traits;

mod util;

pub use crate::optimizers::*;
pub use crate::traits::*;
