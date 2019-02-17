// Third party imports
use ndarray::{Array1};

pub type Func = Fn(&Array1<f64>) -> (f64, f64);

/// Return value and laplacian of the 1s hydrogen orbital
pub fn hydrogen_1s(pos: &Array1<f64>, width: f64) -> (f64, f64) {
    let r = (pos*pos).scalar_sum().sqrt();
    let exp = (-r/width).exp();
    (exp, (1./width.powi(2) - 2./(r*width))*exp)
}

/// Return value and laplacian of the 2s hydrogen orbital
pub fn hydrogen_2s(pos: &Array1<f64>, width: f64) -> (f64, f64) {
    let r = (pos*pos).scalar_sum().sqrt();
    let exp = (-r/width).exp();
    let value = (2. - 2.*r/width)*exp;
    let laplacian = 2.0*exp*(5./width.powi(2) - 4./(width*r) - r/width.powi(3));
    (value, laplacian)
}

/// Return value and Laplacian of Gaussian orbital
pub fn gaussian(pos: &Array1<f64>, width: f64) -> (f64, f64) {
    let r = (pos*pos).scalar_sum().sqrt();
    let width2 = width.powi(2);
    let exp = (-(r).powi(2)/(2.0*width2)).exp();
    (exp, exp/width2*(r.powi(2)/width2 - 3.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Range;
    use ndarray_rand::RandomExt;

    #[test]
    fn hydrogen_ground_state() {
        let r = Array1::<f64>::random(3, Range::new(-1., 1.));
        let wf_val = hydrogen_1s(&r, 1.0).0;
        assert!(wf_val > 0.);
    }

}
