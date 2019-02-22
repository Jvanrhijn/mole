use crate::traits::{BasisSet, Vgl};
use crate::functions::{hydrogen_1s, hydrogen_2s, gaussian};
use ndarray::{FoldWhile, Zip, Array2, Array1};

#[derive(Clone)]
pub struct GaussianBasis {
    centers: Array2<f64>,
    widths: Vec<f64>
}

fn linear_combination_general(pos: &Array1<f64>, coeffs: &Array2<f64>, centers: &Array2<f64>, widths: &Vec<f64>, func: &Fn(&Array1<f64>, f64) -> Vgl) -> Vgl {
    let mut value = 0.0;
    let mut grad = Array1::<f64>::zeros(3);
    let mut laplacian = 0.0;
    Zip::from(centers.genrows())
        .and(coeffs.genrows())
        .apply(|center, coeffs| {
            let (v, g, l) = coeffs.iter().zip(widths.iter())
                .fold((0.0, Array1::<f64>::zeros(3), 0.0) , |acc, (&c, &w)| {
                    let (value, gradient, laplacian) = func(&(pos - &center), w);
                    (acc.0 + c * value, acc.1 + c * &gradient, acc.2 + c * laplacian)
                });
            value += v;
            grad += &g;
            laplacian += l;
        });
    (value, grad, laplacian)
}

impl GaussianBasis {
    pub fn new(centers: Array2<f64>, widths: Vec<f64>) -> Self {
        Self{centers, widths}
    }
}

impl BasisSet for GaussianBasis {
    fn linear_combination(&self, pos: &Array1<f64>, coeffs: &Array2<f64>) -> Vgl {
        linear_combination_general(pos, coeffs, &self.centers, &self.widths, &gaussian)
    }
}

#[derive(Clone)]
pub struct Hydrogen1sBasis {
    centers: Array2<f64>,
    widths: Vec<f64>
}

impl Hydrogen1sBasis {
    pub fn new(centers: Array2<f64>, widths: Vec<f64>) -> Self {
        Self{centers, widths}
    }
}

impl BasisSet for Hydrogen1sBasis {
    fn linear_combination(&self, pos: &Array1<f64>, coeffs: &Array2<f64>) -> Vgl {
        linear_combination_general(pos, coeffs, &self.centers, &self.widths, &hydrogen_1s)
    }
}

#[derive(Clone)]
pub struct Hydrogen2sBasis {
    centers: Array2<f64>,
    widths: Vec<f64>
}

impl Hydrogen2sBasis {
    pub fn new(centers: Array2<f64>, widths: Vec<f64>) -> Self {
        Self{centers, widths}
    }
}

impl BasisSet for Hydrogen2sBasis {
    fn linear_combination(&self, pos: &Array1<f64>, coeffs: &Array2<f64>) -> Vgl {
        linear_combination_general(pos, coeffs, &self.centers, &self.widths, &hydrogen_2s)
    }
}
