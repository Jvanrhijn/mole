// Defines various wave function representations, e.g. Jastrow-Slater
// third party imports
use ndarray::{Array1};
// first party imports
#[allow(unused_imports)]
use traits::wavefunction::*;
use jastrow::JastrowFactor;

#[derive(Debug)]
pub struct JastrowSlater {
    ci_coeffs: Array1<f64>,
    orb_coeffs: Array1<f64>,
    jastrow: JastrowFactor
}

impl JastrowSlater {
    pub fn new(cis: Array1<f64>, orbs: Array1<f64>, jas: JastrowFactor) -> Self {
        JastrowSlater{ci_coeffs: cis, orb_coeffs: orbs, jastrow: jas}
    }
}