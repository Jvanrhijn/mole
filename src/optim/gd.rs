use traits::optimizer::*;
use ndarray::{Array1};

#[allow(dead_code)]
pub struct GradientDescent {
    step_size: f64
}

impl GradientDescent {

    pub fn new(step_size: f64) -> Self {
        GradientDescent{step_size} 
    }

}

impl Optimizer for GradientDescent {
    fn step(&self, parms: &mut Array1<f64>, grads: &Array1<f64>) {
        *parms -= &(grads*self.step_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_gd() {
        let step_size = 0.1;
        let gd = GradientDescent::new(step_size);
        let mut parms = array![1., 2., 3.];
        let grads = array![0.1, 0.2, 0.3];
        gd.step(&mut parms, &grads);
        assert_eq!(parms, array![0.99, 1.98, 2.97]);
    }

}
