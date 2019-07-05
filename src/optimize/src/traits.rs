use errors::Error;
use ndarray::{Array1, Array2};
use operator::OperatorValue;
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, Error>;

pub trait Optimize {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Array1<f64>;

    fn update_parameters(&mut self, deltap: &Array1<f64>);

    fn parameters(&self) -> &Array1<f64>;

    fn num_parameters(&self) -> usize;
}

pub trait Optimizer {
    fn compute_parameter_update(
        &mut self,
        pars: &Array1<f64>,
        averages: &HashMap<String, OperatorValue>,
        raw_data: &HashMap<String, Vec<OperatorValue>>,
    ) -> Result<Array1<f64>>;
}
