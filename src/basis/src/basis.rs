use crate::functions::{gaussian, hydrogen_1s, hydrogen_2s};
use crate::traits::{BasisSet, Vgl};
use ndarray::{Array1, Array2, Zip};

#[derive(Clone)]
pub struct GaussianBasis {
    centers: Array2<f64>,
    widths: Vec<f64>,
}

#[inline]
fn linear_combination_general(
    pos: &Array1<f64>,
    coeffs: &Array2<f64>,
    centers: &Array2<f64>,
    widths: &[f64],
    func: &dyn Fn(&Array1<f64>, f64) -> Vgl,
) -> Vgl {
    let mut value = 0.0;
    let mut grad = Array1::<f64>::zeros(3);
    let mut laplacian = 0.0;
    Zip::from(centers.genrows())
        .and(coeffs.genrows())
        .apply(|center, coeffs| {
            let (v, g, l) = coeffs.iter().zip(widths.iter()).fold(
                (0.0, Array1::<f64>::zeros(3), 0.0),
                |acc, (&c, &w)| {
                    let (value, gradient, laplacian) = func(&(pos - &center), w);
                    (
                        acc.0 + c * value,
                        acc.1 + c * &gradient,
                        acc.2 + c * laplacian,
                    )
                },
            );
            value += v;
            grad += &g;
            laplacian += l;
        });
    (value, grad, laplacian)
}

#[allow(dead_code)]
fn coeff_deriv_general(
    pos: &Array1<f64>,
    coeffs: &Array2<f64>,
    centers: &Array2<f64>,
    widths: &[f64],
    func: &dyn Fn(&Array1<f64>, f64) -> Vgl,
) -> Array2<f64> {
    let shape = coeffs.shape();
    let n = shape[0];
    let m = shape[1];
    let mut grad = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let center = centers.slice(s![i, ..]);
            let (v, _, _) = func(&(pos - &center), widths[j]);
            grad[[i, j]] = v;
        }
    }
    grad
}

impl GaussianBasis {
    pub fn new(centers: Array2<f64>, widths: Vec<f64>) -> Self {
        Self { centers, widths }
    }
}

impl BasisSet for GaussianBasis {
    fn linear_combination(&self, pos: &Array1<f64>, coeffs: &Array2<f64>) -> Vgl {
        linear_combination_general(pos, coeffs, &self.centers, &self.widths, &gaussian)
    }

    fn coefficient_derivative(&self, pos: &Array1<f64>, coeffs: &Array2<f64>) -> Array2<f64> {
        coeff_deriv_general(pos, coeffs, &self.centers, &self.widths, &gaussian)
    }
}

#[derive(Clone)]
pub struct Hydrogen1sBasis {
    centers: Array2<f64>,
    widths: Vec<f64>,
}

impl Hydrogen1sBasis {
    pub fn new(centers: Array2<f64>, widths: Vec<f64>) -> Self {
        Self { centers, widths }
    }
}

impl BasisSet for Hydrogen1sBasis {
    fn linear_combination(&self, pos: &Array1<f64>, coeffs: &Array2<f64>) -> Vgl {
        linear_combination_general(pos, coeffs, &self.centers, &self.widths, &hydrogen_1s)
    }

    fn coefficient_derivative(&self, pos: &Array1<f64>, coeffs: &Array2<f64>) -> Array2<f64> {
        coeff_deriv_general(pos, coeffs, &self.centers, &self.widths, &hydrogen_1s)
    }
}

#[derive(Clone)]
pub struct Hydrogen2sBasis {
    centers: Array2<f64>,
    widths: Vec<f64>,
}

impl Hydrogen2sBasis {
    pub fn new(centers: Array2<f64>, widths: Vec<f64>) -> Self {
        Self { centers, widths }
    }
}

impl BasisSet for Hydrogen2sBasis {
    fn linear_combination(&self, pos: &Array1<f64>, coeffs: &Array2<f64>) -> Vgl {
        linear_combination_general(pos, coeffs, &self.centers, &self.widths, &hydrogen_2s)
    }

    fn coefficient_derivative(&self, pos: &Array1<f64>, coeffs: &Array2<f64>) -> Array2<f64> {
        coeff_deriv_general(pos, coeffs, &self.centers, &self.widths, &hydrogen_2s)
    }
}
