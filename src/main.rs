#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate rand;
extern crate assert;

use std::vec::Vec;
use rand::random;
use ndarray::{Array1, arr2, Axis, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Range;

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
    pub mod basis;
}

mod metropolis;
mod wf;
mod jastrow;
mod orbitals;
mod determinant;
mod operators;
mod error;

fn main() {
    // number of electrons
    let nelec = 2;

    // create basis function set
    let basis_set: Vec<Box<Fn(&Array1<f64>) -> (f64, f64)>> = vec![
        Box::new(|x| hydrogen_1s(&x)),
        Box::new(|x| hydrogen_2s(&x))
    ];

    // create orbitals from basis functions
    let orbital1 = Orbital::new(array![1.0, 0.0], &basis_set);
    let orbital2 = Orbital::new(array![0.0, 1.0], &basis_set);

    // Initialize wave function: single Slater determinant
    let mut wf = wf::SingleDeterminant::new(vec![orbital1, orbital2]);

    // setup Hamiltonian components
    let v = IonicPotential::new(array![[0., 0., 0.]], array![1]);
    let t = KineticEnergy::new();
    let ve = ElectronicPotential::new();

    // setup electronic structure Hamiltonian
    let h = ElectronicHamiltonian::new(t, v, ve);

    // max number of MC steps and equilibration time
    let iters = 1000usize;
    let equib = iters/10;

    // initial random coanfiguration
    let mut cfg = Array2::<f64>::random((nelec, 3), Range::new(-1., 1.));

    // initialize wave function
    wf.update(&cfg);

    // vector for storing local energy
    let mut local_energy = Vec::<f64>::new();

    // acceptance rate
    let mut acceptance = 0usize;

    // QMC loop
    for i in 0..iters {
        // move each electron in turn
        for j in 0..nelec {
            // propose a move, if accepted: update the wave function
            // else, keep the same wave function
            if let Some(config) = metropolis::metropolis_single_move_box(&wf, &cfg, j) {
                cfg = config;
                wf.update(&cfg);
                acceptance += 1;
            }
            // calculate local energy: Eloc = H(\psi)/(\psi)
            let hpsi = h.act_on(&wf, &cfg).expect("Failed to act on \\psi with H");
            let local_e = hpsi/wf.value(&cfg).unwrap();
            // save local energy, discard if we get NaN,
            // discard non-equilibrated values
            if !local_e.is_nan() && i > equib {
                println!("Local E = {:.*}", 5, local_e);
                local_energy.push(local_e);
            }
        }
    }

    // calculate final values (means etc)
    let local_energy = Array1::<f64>::from_vec(local_energy);

    let mean_local_energy = local_energy.mean_axis(Axis(0)).scalar_sum();
    let std_local_energy = local_energy.var_axis(Axis(0), 0.).scalar_sum().sqrt();

    println!("Final local E: {:.*} +/- {:.*}", 5, mean_local_energy, 10, std_local_energy);
    println!("Acceptance rate: {}", acceptance as f64 / (iters*nelec) as f64);

}
