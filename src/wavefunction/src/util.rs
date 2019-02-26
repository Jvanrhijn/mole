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
        let mut cfg_plus = cfg.clone();
        let mut cfg_minus = cfg.clone();
        for k in 0..3 {
            cfg_plus[[row, k]] += step;
            cfg_minus[[row, k]] -= step;
            let func_plus = func.value(&cfg_plus)?;
            let func_minus = func.value(&cfg_minus)?;
            grad[[row, k]] = (func_plus - func_minus) / (2.0 * step);
            laplac += (func_plus - 2.0 * f + func_minus) / step.powi(2);
            cfg_plus[[row, k]] -= step;
            cfg_minus[[row, k]] += step;
        }
    }
    Ok((grad, laplac))
}
