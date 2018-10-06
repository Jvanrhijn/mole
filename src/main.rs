mod optim {
    pub mod gd;
}

fn main() {
    let g = optim::gd::GradientDescent::new(0.1);
}
