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
use operators::*;
use traits::operator::*;

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
mod operators;


fn main() {

    let basis_set: Vec<Box<Fn(&Array1<f64>) -> (f64, f64)>> = vec![Box::new(hydrogen_1s), Box::new(hydrogen_2s)];

    let orbital = OrbitalExact::new(array![1.0, 0.0], &basis_set);
    let wf = wf::SingleDeterminant::new(vec![orbital]);
    let mut cfg = arr2(&[[-1., 0., 0.]]);

    let v = IonicPotential::new(array![[0.0, 0., 0.0]], array![1]);
    let t = KineticEnergy::new();

    let h = IonicHamiltonian::new(t, v);

    match metropolis::metropolis_single_move_box(&wf, &cfg, 0) {
        Some(config) => { println!("Move accepted"); cfg = config }
        None         => { println!("Move rejected") }
    }

    let local_energy = h.act_on(&wf, &cfg)/wf.value(&cfg).unwrap();

    println!("Local E: {}", local_energy);

}
