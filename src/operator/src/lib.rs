#![feature(uniform_paths)]
#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;

mod traits;
mod operator;

pub use crate::traits::*;
pub use crate::operator::*;
