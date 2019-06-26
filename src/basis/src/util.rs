use ndarray::Array1;

type Vgl = (f64, Array1<f64>, f64);

#[allow(dead_code)]
pub fn grad_laplacian_finite_difference(
    func: &dyn Fn(&Array1<f64>) -> Vgl,
    cfg: &Array1<f64>,
    step: f64,
) -> (Array1<f64>, f64) {
    let mut grad = Array1::zeros(3);
    let mut laplac = 0.0;
    let f = func(cfg).0;
    for k in 0..3 {
        let mut cfg_plus = cfg.clone();
        let mut cfg_minus = cfg.clone();
        let mut cfg_plus2 = cfg.clone();
        let mut cfg_minus2 = cfg.clone();
        cfg_plus[k] += step;
        cfg_plus2[k] += 2.0 * step;
        cfg_minus[k] -= step;
        cfg_minus2[k] -= 2.0 * step;
        let func_plus = func(&cfg_plus).0;
        let func_plus2 = func(&cfg_plus2).0;
        let func_minus = func(&cfg_minus).0;
        let func_minus2 = func(&cfg_minus2).0;
        grad[k] = (-func_plus2 + 8.0 * func_plus - 8.0 * func_minus + func_minus2) / (12.0 * step);
        laplac += (-func_plus2 + 16.0 * func_plus - 30.0 * f + 16.0 * func_minus - func_minus2)
            / (12.0 * step.powi(2));
    }
    (grad, laplac)
}
