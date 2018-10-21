use ndarray::{Array1, Array, Ix2};
use traits::function::*;
use error::{Error};

/// Jastrow factor struct, used to construct Jastrow-Slater wave functions.
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

    fn value(&self, _cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        // Stub, TODO implement actual Jastrow factor
        Ok(1.0)
    }
}