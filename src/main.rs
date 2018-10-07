#[macro_use]
extern crate ndarray;

mod optim {
    pub mod gd;
}

mod traits {
    pub mod optimizer;
    pub mod wavefunction;
}

mod wf;

fn main() {
    let wf = wf::JastrowSlater::new(array![1., 2., 3.], array![1., 2., 3.], array![1., 2., 3.]);
    let _g = optim::gd::GradientDescent::new(0.1);
    println!("{:?}", wf);
}
