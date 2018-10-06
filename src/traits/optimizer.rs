pub trait Optimizer {
    fn step(&self, parms: &mut [f64], grads: &[f64]);
}
