#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate rand;
extern crate assert;

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
    let basis_set = vec![basis_sets::Hydrogen1s{}];
    let basis_set2 = vec![basis_sets::Hydrogen2s{}];
    let orbital = orbitals::OrbitalExact::new(ndarray::Array1::from_vec(vec![1.]), basis_set);
    let orbital2 = orbitals::OrbitalExact::new(ndarray::Array1::from_vec(vec![1.]), basis_set2);
    let wf = wf::SingleDeterminant::new(vec![orbital, orbital2]);
    let cfg = ndarray::Array2::<f64>::ones((2, 3));
    println!("{}", wf.value(&cfg).unwrap())
}
