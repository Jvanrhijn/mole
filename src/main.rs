#![deny(clippy::all, clippy::pedantic)]
#![allow(dead_code)]
#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate rand;
extern crate num_traits;

mod optim {
    pub mod gd;
}

mod traits {
    pub mod optimizer;
    pub mod differentiate;
    pub mod function;
    pub mod operator;
    pub mod metropolis;
    pub mod mcsamplers;
    pub mod wavefunction;
}

mod math {
    pub mod basis;
}

mod metrop;
mod block;
mod wf;
mod jastrow;
mod orbitals;
mod determinant;
mod operators;
mod error;
mod montecarlo;

use std::vec::Vec;
use ndarray::{Array1};

use math::basis::*;
use orbitals::*;
use operators::*;
use montecarlo::{Sampler, Runner};


fn hydrogen_molecule_demo() {
    // create basis function set
    let basis_set: Vec<Box<Fn(&Array1<f64>) -> (f64, f64)>> = vec![
        Box::new(|x| hydrogen_1s(&(x + &array![1.0, 0., 0.]))),
        Box::new(|x| hydrogen_1s(&(x - &array![1.0, 0., 0.])))
    ];

    // create orbitals from basis functions
    let orbital1 = Orbital::new(array![1.0, 0.0], &basis_set);
    let orbital2 = Orbital::new(array![0.0, 1.0], &basis_set);

    // Initialize wave function: single Slater determinant
    let mut wf = wf::SingleDeterminant::new(vec![orbital1, orbital2]);

    // setup Hamiltonian components
    let v = IonicPotential::new(array![[-1., 0., 0.], [1.0, 0., 0.]], array![1, 1]);
    let t = KineticEnergy::new();
    let ve = ElectronicPotential::new();

    // setup electronic structure Hamiltonian
    let local_e = LocalEnergy::new(ElectronicHamiltonian::new(t, v, ve));

    // create metropolis algorithm
    let metrop = metrop::MetropolisBox::new(1.0);

    // setup monte carlo sampler
    let mut sampler = Sampler::new(&mut wf, metrop);
    sampler.add_observable(local_e);

    // create runner
    let mut runner = Runner::new(sampler);

    runner.run(100, 1000);

    println!("Local E:       {:.*}", 5, runner.means()[0]);
}

fn main() {
    hydrogen_molecule_demo();
}
