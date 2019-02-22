use ndarray::{Array1, Array2};
use std::ops::Add;

pub type Vgl = (f64, Array1<f64>, f64);

pub trait BasisSet {
    fn linear_combination(&self, pos: &Array1<f64>, coeffs: &Array2<f64>) -> Vgl;
}