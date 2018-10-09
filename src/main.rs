#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_linalg;
extern crate rand;
extern crate assert;

mod optim {
    pub mod gd;
}

mod traits {
    pub mod optimizer;
    pub mod wavefunction;
    pub mod orbital;
    pub mod function;
}

mod math {
    pub mod mat_ops;
}

mod metropolis;
mod wf;
mod jastrow;
mod orbitals;
mod determinant;

fn main() {
    let jas = jastrow::JastrowFactor::new(array![1.], array![1.], array![1.]);
    let wf = wf::JastrowSlater::new(array![1., 2., 3.], array![1., 2., 3.], jas);
    let _g = optim::gd::GradientDescent::new(0.1);
    println!("{:?}", wf);
}
