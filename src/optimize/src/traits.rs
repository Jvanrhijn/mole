use ndarray::{Array1, Array2};

pub trait Optimize {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Array1<f64>;

    fn update_parameters(&mut self, deltap: &Array1<f64>);

    fn parameters(&self) -> &Array1<f64>;

    fn num_parameters(&self) -> usize;
}

pub trait Optimizer {
    // TODO refactor into taking all MC data
    // now takes: energy gradient, wave function values,
    // parameter gradients, parameter values
    fn compute_parameter_update(
        &mut self,
        ud: &(Array1<f64>, Array1<f64>, Array2<f64>, Array1<f64>),
    ) -> Array1<f64>;
}
