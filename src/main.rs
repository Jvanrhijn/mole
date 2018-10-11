#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate rand;
extern crate assert;

use std::vec::Vec;
use ndarray::{Array1, arr2};

use traits::function::Function;
use math::basis::*;
use orbitals::*;

mod optim {
    pub mod gd;
}

mod traits {
    pub mod optimizer;
    pub mod wavefunction;
    pub mod function;
    pub mod operator;
}

mod math {
    pub mod mat_ops;
    pub mod basis;
}

mod metropolis;
mod wf;
mod jastrow;
mod orbitals;
mod determinant;
mod basis_sets;

fn main() {
    let mut basis_set = Vec::<&Fn(&Array1<f64>) -> f64>::new();
    basis_set.push(&hydrogen_1s);
    basis_set.push(&hydrogen_2s);

    let orbital1 = OrbitalExact::new(array![2f64.sqrt(), 2f64.sqrt()], &basis_set);
    let orbital2 = OrbitalExact::new(array![2f64.sqrt(), -2f64.sqrt()], &basis_set);

    let wf = wf::SingleDeterminant::new(vec![orbital1, orbital2]);
    let cfg = arr2(&[[1., -1., 0.], [-1., 1., 1.]]);

    println!("{}", wf.value(&cfg).unwrap())
}
