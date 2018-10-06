// Defines various wave function representations, e.g. Jastrow-Slater
use std::vec::{Vec};
use traits::wavefunction::*;

pub struct JastrowSlater {
    ci_coeffs: Vec<f64>,
    orb_coeffs: Vec<f64>,
    jas_coeffs: Vec<f64>
}

impl JastrowSlater {
    pub fn new(cis: &[f64], orbs: &[f64], jas: &[f64]) -> Self {
        JastrowSlater{ci_coeffs: cis.to_vec(), orb_coeffs: orbs.to_vec(), jas_coeffs: jas.to_vec()}
    }
}