#![deny(clippy::all, clippy::pedantic)]
#![allow(dead_code)]
#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate rand;

mod optim {
    pub mod gd;
}

mod traits {
    pub mod optimizer;
    pub mod wavefunction;
    pub mod function;
    pub mod operator;
    pub mod metropolis;
    pub mod mcsamplers;
}

mod math {
    pub mod basis;
}

mod metrop;
mod wf;
mod jastrow;
mod orbitals;
mod determinant;
mod operators;
mod error;
mod montecarlo;

use std::vec::Vec;
use ndarray::{Array1, Axis, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Range;

use traits::function::Function;
use traits::mcsamplers::MonteCarloSampler;
use math::basis::*;
use orbitals::*;
use operators::*;
use traits::operator::*;
use traits::metropolis::Metropolis;
use montecarlo::{Sampler, Runner};

fn main() {
    // number of electrons
    let nelec = 2;
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
    let v = IonicPotential::new(array![[-1., 0., 0.], [1., 0., 0.]], array![1, 1]);
    let t = KineticEnergy::new();
    let ve = ElectronicPotential::new();

    // setup electronic structure Hamiltonian
    let local_e = LocalEnergy::new(ElectronicHamiltonian::new(t, v, ve));

    // max number of MC steps and equilibration time
    let iters = 1000_usize;

    // initial random configuration
    let mut cfg = Array2::<f64>::random((nelec, 3), Range::new(-1., 1.))*(nelec as f64);

    // create metropolis algorithm
    let mut metrop = metrop::MetropolisBox::new(1.0, wf.value(&cfg).unwrap());

    // setup monte carlo sampler
    let mut sampler = Sampler::new(&mut wf, metrop);
    sampler.add_observable(local_e);

    // create runner
    let mut runner = Runner::new(sampler);

    runner.run(iters);

    let means = runner.means();

    println!("Local E: {}", means[0]);

}
