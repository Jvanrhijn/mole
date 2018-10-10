#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate rand;
extern crate assert;

use std::vec::Vec;
use ndarray::Array1;

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
    let mut basis_set = Vec::<&Fn(&Array1<f64>) -> f64>::new();
    basis_set.push(&math::basis::hydrogen_1s);
    basis_set.push(&math::basis::hydrogen_2s);

    let orbital = orbitals::OrbitalExact::new(ndarray::Array1::from_vec(vec![1., 1.]), basis_set);
    let wf = wf::SingleDeterminant::new(vec![orbital]);
    let cfg = ndarray::Array2::<f64>::ones((2, 3));
    println!("{}", wf.value(&cfg).unwrap())
}
