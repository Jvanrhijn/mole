use ndarray::{Array1, Array, Ix2};
use traits::function::*;

#[derive(Debug)]
pub struct JastrowFactor {
    parms_one_body: Array1<f64>,
    parms_two_body: Array1<f64>,
    parms_three_body: Array1<f64>

}

impl JastrowFactor {
    pub fn new(parms_one_body: Array1<f64>,
               parms_two_body: Array1<f64>,
               parms_three_body: Array1<f64>) -> Self {
        JastrowFactor{parms_one_body, parms_two_body, parms_three_body}
    }

}

impl Function<f64> for JastrowFactor {

    type D = Ix2;

    fn value(&self, cfg: &Array<f64, Self::D>) -> f64 {
        // Stub, TODO implement actual Jastrow factor
        1.0
    }
}