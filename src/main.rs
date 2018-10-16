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

    let basis_set: Vec<Box<Fn(&Array1<f64>) -> (f64, f64)>> = vec![
        Box::new(|x| gaussian(&(x + &array![1., 0., 0.]), 0.1)),
        Box::new(|x| hydrogen_1s(&(x - &array![1., 0., 0.])))
    ];

    let orbital = OrbitalExact::new(array![1.0, 0.0], &basis_set);
    let wf = wf::SingleDeterminant::new(vec![orbital]);

    let v = IonicPotential::new(array![[-1., 0., 0.], [1., 0., 0.]], array![1, 0]);
    let t = KineticEnergy::new();
    let ve = ElectronicPotential::new();
    let h = ElectronicHamiltonian::new(t, v, ve);

    let iters = 10000usize;

    let mut cfg = Array2::<f64>::random((1, 3), Range::new(-1., 1.));
    let mut local_energy = Vec::<f64>::new();
    let mut acceptance = 0usize;

    for i in 0..iters {
        match metropolis::metropolis_single_move_box(&wf, &cfg, 0) {
            Some(config) => {
                cfg = config;
                acceptance += 1usize;
            }
            None => ()
        }

        local_energy.push(h.act_on(&wf, &cfg)/wf.value(&cfg).unwrap());
    }

    let local_energy = Array1::<f64>::from_vec(local_energy);

    let mean_local_energy = local_energy.mean_axis(Axis(0)).scalar_sum();
    let std_local_energy = local_energy.var_axis(Axis(0), 0.).scalar_sum().sqrt();

    println!("Local E: {:.*} +/- {:.*}", 5, mean_local_energy, 10, std_local_energy);
    println!("Acceptance rate: {}", acceptance as f64 / iters as f64);

}
