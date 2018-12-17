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

fn get_hydrogen_runner(basis_set: &Vec<Box<Func>>)
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
    let v = IonicPotential::new(array![[-1., 0., 0.], [1., 0., 0.]], array![1, 1]);
    let t = KineticEnergy::new();
    let ve = ElectronicPotential::new();

    // setup electronic structure Hamiltonian
    let local_e = LocalEnergy::new(ElectronicHamiltonian::new(t, v, ve));

    // create metropolis algorithm
    let metrop = metrop::MetropolisBox::from_rng(1., rng);

    // setup monte carlo sampler
    let mut sampler = Sampler::new( wf, metrop);
    sampler.add_observable(local_e);

    // create runner
    Runner::new(sampler)
}

fn blocking_analysis() {
    let basis_set: Vec<Box<Func>> = vec![
        Box::new(|x| hydrogen_2s(&(x + &array![1.0, 0., 0.]))),
        Box::new(|x| hydrogen_2s(&(x - &array![1.0, 0., 0.])))
    ];

    let num_steps = 2_usize.pow(14);

    let mut last_block_size = 0;
    for num_blocks in (20..num_steps/2).step_by(num_steps/500) {
        let block_size = num_steps/num_blocks;
        if last_block_size == block_size {
            continue;
        }
        last_block_size = block_size;
        let mut runner = get_hydrogen_runner(&basis_set);
        runner.run(num_steps, block_size);

        let local_e = runner.means()[0];
        let stdev= runner.variances()[0].sqrt();
        let stdev_error = stdev*1.0/(2.0*(block_size - 1) as f64).sqrt();

        println!("Local E: {:.*} stdev: {:.*} +/- {:.*} {}", 16, local_e, 16, stdev, 16, stdev_error, num_blocks);
    }

}

fn main() {
    blocking_analysis();
}
