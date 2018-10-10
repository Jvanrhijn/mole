#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate rand;
extern crate assert;

use std::vec::Vec;
use ndarray::{Array1, arr2};

use traits::function::Function;

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
    let mut basis_set_first = Vec::<&Fn(&Array1<f64>) -> f64>::new();
    let mut basis_set_second = Vec::<&Fn(&Array1<f64>) -> f64>::new();
    basis_set_first.push(&math::basis::hydrogen_1s);
    basis_set_second.push(&math::basis::hydrogen_2s);

    let orbital1 = orbitals::OrbitalExact::new(array![1.], basis_set_first);
    let orbital2 = orbitals::OrbitalExact::new(array![1.], basis_set_second);

    let wf = wf::SingleDeterminant::new(vec![orbital1, orbital2]);
    let cfg = arr2(&[[1., -1., 0.], [-1., 1., 1.]]);
    println!("{}", wf.value(&cfg).unwrap())
}
