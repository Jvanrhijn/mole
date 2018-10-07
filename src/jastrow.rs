use ndarray::{Array1};

pub struct JastrowFactor<'a> {
    parms: &'a mut Array1<f64>,
}