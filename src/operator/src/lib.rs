#![feature(exclusive_range_pattern)]
#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;

mod operator;
mod traits;

pub use crate::operator::*;
pub use crate::traits::*;
