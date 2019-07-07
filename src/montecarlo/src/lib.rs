extern crate itertools;
#[allow(unused_imports)]
#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;

pub mod montecarlo;
pub mod samplers;
pub mod traits;

pub use crate::montecarlo::*;
pub use crate::samplers::*;
