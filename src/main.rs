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


fn main() {
    let wf_1s_left = |x: &Array1<f64>| hydrogen_1s(&(x - &array![-0.5, 0., 0.]));
    let wf_1s_right = |x: &Array1<f64>| hydrogen_2s(&(x - &array![0.5, 0., 0.]));

    let basis_set: Vec<Box<Fn(&Array1<f64>) -> f64>> = vec![
        Box::new(wf_1s_left),
        Box::new(wf_1s_right),
    ];

    let orbital1 = OrbitalExact::new(array![1., 1.], &basis_set);
    let orbital2 = OrbitalExact::new(array![1., -1.], &basis_set);

    let wf = wf::SingleDeterminant::new(vec![orbital1, orbital2]);
    let mut cfg = arr2(&[[1., -1., 0.], [-1., 1., 1.]]);

    println!("{}", wf.value(&cfg).unwrap());

    match metropolis::metropolis_single_move_box(&wf, &cfg) {
        Some(config) => { println!("Move accepted"); cfg = config }
        None         => { println!("Move rejected") }
    }

    println!("{}", wf.value(&cfg).unwrap());
}
