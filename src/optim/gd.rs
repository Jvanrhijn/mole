use traits::optimizer::*;

pub struct GradientDescent {
    step_size: f64
}

impl GradientDescent {

    pub fn new(step_size: f64) -> Self {
        GradientDescent{step_size} 
    }

}

impl Optimizer for GradientDescent {
    fn step(&self, parms: &mut [f64], grads: &[f64]) {
        for (parm, grad) in parms.iter_mut().zip(grads.iter()) {
            *parm -= self.step_size*(*grad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_gd() {
        let step_size = 0.1;
        let gd = GradientDescent::new(step_size);
        let mut parms = vec![1., 2., 3.];
        let grads = vec![0.1, 0.2, 0.3];
        gd.step(parms.as_mut_slice(), grads.as_slice());
        assert_eq!(parms, vec![0.99, 1.98, 2.97]);
    }

}
