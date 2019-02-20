#![feature(uniform_paths)]
extern crate itertools;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;

mod montecarlo;
mod block;
mod samplers;
mod traits;

pub use crate::montecarlo::*;
pub use crate::samplers::*;
