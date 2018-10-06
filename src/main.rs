mod optim {
    pub mod gd;
}

mod traits {
    pub mod optimizer;
}

fn main() {
    let g = optim::gd::GradientDescent::new(0.1);
}
