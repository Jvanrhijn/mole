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
    let basis_set = vec![basis_sets::HydrogenGroundState{}];
    let orbital = orbitals::OrbitalExact::new(ndarray::Array1::from_vec(vec![1.]), basis_set);
    let wf = wf::SingleDeterminant::new(vec![orbital]);
    let cfg = ndarray::Array2::<f64>::ones((1, 3));
    println!("{}", wf.value(&cfg).unwrap())
}
