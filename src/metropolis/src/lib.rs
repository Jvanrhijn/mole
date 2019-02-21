#![feature(uniform_paths)]
#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;

mod traits;
mod metrop;

pub use traits::*;
pub use metrop::*;
