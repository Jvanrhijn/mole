// Third party imports
use ndarray::{Array1};

pub fn hydrogen_1s(pos: &Array1<f64>) -> f64 {
    let r = (pos*pos).scalar_sum().sqrt();
    (-r).exp()
}

pub fn hydrogen_2s(pos: &Array1<f64>) -> f64 {
    let r = (pos*pos).scalar_sum().sqrt();
    (1. - r)*((-r).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{random,  distributions::Range};
    use ndarray_rand::RandomExt;

    #[test]
    fn ground_state() {
        let r = Array1::<f64>::random(3, Range::new(-1., 1.));
        let wf_val = hydrogen_1s(&r);
        assert!(wf_val > 0.);
    }
}
