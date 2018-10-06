// Defines various wave function representations, e.g. Jastrow-Slater
// standard imports
use std::vec::{Vec};
// third party imports
use ndarray::{Array, Array1};
// first party imports
use traits::wavefunction::*;

#[derive(Debug)]
pub struct JastrowSlater {
    ci_coeffs: Array1<f64>,
    orb_coeffs: Array1<f64>,
    jas_coeffs: Array1<f64>
}

impl JastrowSlater {
    pub fn new(cis: Array1<f64>, orbs: Array1<f64>, jas: Array1<f64>) -> Self {
        JastrowSlater{ci_coeffs: cis, orb_coeffs: orbs, jas_coeffs: jas}
    }
}