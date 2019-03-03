use crate::error::Error;
use crate::traits::{Differentiate, Function};
use ndarray::{Array2, Ix2};

#[allow(dead_code)]
pub fn grad_laplacian_finite_difference<T>(
    func: &T,
    cfg: &Array2<f64>,
    step: f64,
) -> Result<(Array2<f64>, f64), Error>
where
    T: Function<f64, D = Ix2> + Differentiate,
{
    let mut grad = Array2::zeros((cfg.shape()[0], cfg.shape()[1]));
    let mut laplac = 0.0;
    let f = func.value(cfg)?;
    for row in 0..cfg.shape()[0] {
        for k in 0..3 {
            let mut cfg_plus = cfg.clone();
            let mut cfg_minus = cfg.clone();
            let mut cfg_plus2 = cfg.clone();
            let mut cfg_minus2 = cfg.clone();
            cfg_plus[[row, k]] += step;
            cfg_plus2[[row, k]] += 2.0 * step;
            cfg_minus[[row, k]] -= step;
            cfg_minus2[[row, k]] -= 2.0 * step;
            let func_plus = func.value(&cfg_plus)?;
            let func_plus2 = func.value(&cfg_plus2)?;
            let func_minus = func.value(&cfg_minus)?;
            let func_minus2 = func.value(&cfg_minus2)?;
            grad[[row, k]] =
                (-func_plus2 + 8.0 * func_plus - 8.0 * func_minus + func_minus2) / (12.0 * step);
            laplac += (-func_plus2 + 16.0 * func_plus - 30.0 * f + 16.0 * func_minus - func_minus2)
                / (12.0 * step.powi(2));
        }
    }
    Ok((grad, laplac))
}
