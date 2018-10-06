pub trait Optimizer {
    fn step(&mut self, parms: &mut [f64], grads: &[f64]);
}
