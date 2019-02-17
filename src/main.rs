#![deny(clippy::all, clippy::pedantic)]
#![allow(dead_code)]
#[macro_use]
extern crate ndarray;

use std::env;
use rand::{StdRng, SeedableRng};

use wavefunction::{SingleDeterminant, Orbital};
use metropolis::MetropolisBox;
use basis::*;
use montecarlo::{Runner, Sampler};
use operator::*;


fn get_hydrogen2_runner(basis_set: &Vec<Box<Func>>, box_size: f64)
    -> Runner<Sampler<SingleDeterminant<Func>, MetropolisBox<StdRng>>>
{
    // create seeded rng
    let seed: [u8; 32] = [0; 32];
    let rng = StdRng::from_seed(seed);

    // create orbitals from basis functions
    let orbital1 = Orbital::new(array![1.0, 1.0], basis_set);
    let orbital2 = Orbital::new(array![1.0, -1.0], basis_set);

    // Initialize wave function: single Slater determinant
    let wf = SingleDeterminant::new(vec![orbital1, orbital2]);

    // setup Hamiltonian components
    let v = IonicPotential::new(array![[-1.0, 0., 0.], [1.0, 0., 0.]], array![1, 1]);
    let t = KineticEnergy::new();
    let ve = ElectronicPotential::new();

    // setup electronic structure Hamiltonian
    let local_e = LocalEnergy::new(ElectronicHamiltonian::new(t, v, ve));

    // create metropolis algorithm
    let metrop = MetropolisBox::from_rng(box_size, rng);

    // setup monte carlo sampler
    let mut sampler = Sampler::new( wf, metrop);
    sampler.add_observable("Local Energy", local_e);

    // create runner
    Runner::new(sampler)
}


fn get_hydrogen_runner(basis_set: &Vec<Box<Func>>, box_size: f64) 
    -> Runner<Sampler<SingleDeterminant<Func>, MetropolisBox<StdRng>>>
{
    let seed: [u8; 32] = [0; 32];
    let rng = StdRng::from_seed(seed);

    let orbital = Orbital::new(array![1.0], basis_set);

    let wf = SingleDeterminant::new(vec![orbital]);

    let v = IonicPotential::new(array![[0.0, 0.0, 0.0]], array![1]);
    let t = KineticEnergy::new();
    let ve = ElectronicPotential::new();

    let local_e = LocalEnergy::new(ElectronicHamiltonian::new(t, v, ve));

    let metrop = MetropolisBox::from_rng(box_size, rng);

    let mut sampler = Sampler::new(wf, metrop);
    sampler.add_observable("Local Energy", local_e);

    Runner::new(sampler)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let box_side = args[args.len()-1].parse::<f64>().unwrap();

    //let _basis_set: Vec<Box<Func>> = vec![
    //    Box::new(|x| hydrogen_1s(&(x + &array![1.0, 0., 0.]), 1.0)),
    //    Box::new(|x| hydrogen_1s(&(x - &array![1.0, 0., 0.]), 1.0))
    //];

    let gauss_basis: Vec<Box<Func>> = vec![
        Box::new(|x| gaussian(x, 1.0))
    ];

    let num_steps = 1_000_000;

    let mut runner = get_hydrogen_runner(&gauss_basis, box_side);
    runner.run(num_steps, 1);
}
