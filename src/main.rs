mod optim {
    pub mod gd;
}

mod traits {
    pub mod optimizer;
    pub mod wavefunction;
}

mod wf;

fn main() {
    let wf = wf::JastrowSlater::new(&[1., 2., 3.], &[1., 2., 3.], &[1., 2., 3.]);
    let g = optim::gd::GradientDescent::new(0.1);
}
