#![deny(clippy::all, clippy::pedantic)]
#![allow(dead_code)]
#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate rand;
extern crate num_traits;
#[macro_use]
extern crate itertools;
use std::env;

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
    pub mod cache;
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
use rand::rngs::SmallRng;
use rand::SeedableRng;

use math::basis::*;
use orbitals::*;
use operators::*;
use montecarlo::{Sampler, Runner};

type Func = Fn(&Array1<f64>) -> (f64, f64);

fn get_hydrogen2_runner(basis_set: &Vec<Box<Func>>, box_size: f64)
    -> Runner<Sampler<wf::SingleDeterminant<Func>, metrop::MetropolisBox<SmallRng>>>
{
    // create seeded rng
    let rng = SmallRng::seed_from_u64(0);

    // create orbitals from basis functions
    let orbital1 = Orbital::new(array![1.0, 1.0], basis_set);
    let orbital2 = Orbital::new(array![1.0, -1.0], basis_set);

    // Initialize wave function: single Slater determinant
    let wf = wf::SingleDeterminant::new(vec![orbital1, orbital2]);

    // setup Hamiltonian components
    let v = IonicPotential::new(array![[-1.0, 0., 0.], [1.0, 0., 0.]], array![1, 1]);
    let t = KineticEnergy::new();
    let ve = ElectronicPotential::new();

    // setup electronic structure Hamiltonian
    let local_e = LocalEnergy::new(ElectronicHamiltonian::new(t, v, ve));

    // create metropolis algorithm
    let metrop = metrop::MetropolisBox::from_rng(box_size, rng);

    // setup monte carlo sampler
    let mut sampler = Sampler::new( wf, metrop);
    sampler.add_observable(local_e);

    // create runner
    Runner::new(sampler)
}


fn get_hydrogen_runner(basis_set: &Vec<Box<Func>>, box_size: f64) 
    -> Runner<Sampler<wf::SingleDeterminant<Func>, metrop::MetropolisBox<SmallRng>>>
{
    let rng = SmallRng::seed_from_u64(0);

    let orbital = Orbital::new(array![1.0], basis_set);

    let wf = wf::SingleDeterminant::new(vec![orbital]);

    let v = IonicPotential::new(array![[0.0, 0.0, 0.0]], array![1]);
    let t = KineticEnergy::new();
    let ve = ElectronicPotential::new();

    let local_e = LocalEnergy::new(ElectronicHamiltonian::new(t, v, ve));

    let metrop = metrop::MetropolisBox::from_rng(box_size, rng);

    let mut sampler = Sampler::new(wf, metrop);
    sampler.add_observable(local_e);

    Runner::new(sampler)
}


fn main() {
    let args: Vec<String> = env::args().collect();
    let box_side = args[args.len()-1].parse::<f64>().unwrap();

    let basis_set: Vec<Box<Func>> = vec![
        Box::new(|x| hydrogen_1s(&(x + &array![1.0, 0., 0.]))),
        Box::new(|x| hydrogen_1s(&(x - &array![1.0, 0., 0.])))
    ];

    let gauss_basis: Vec<Box<Func>> = vec![
        Box::new(|x| gaussian(x, 5.0))
    ];

    let num_steps = 1_000_000;

    //let mut runner = get_hydrogen2_runner(&basis_set, box_side);
    let mut runner = get_hydrogen_runner(&gauss_basis, box_side);
    runner.run(num_steps, 1);
}
