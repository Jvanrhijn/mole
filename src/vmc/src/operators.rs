use errors::Error;
use ndarray::Array2;
use operator::{Operator, OperatorValue};
use optimize::Optimize;
use wavefunction_traits::Cache;

pub struct ParameterGradient;

impl<T: Optimize + Cache> Operator<T> for ParameterGradient {
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue, Error> {
        Ok(OperatorValue::Vector(wf.parameter_gradient(cfg)?)
            * OperatorValue::Scalar(wf.current_value()?.0))
    }
}

#[derive(Copy, Clone)]
pub struct WavefunctionValue;

impl<T: Cache> Operator<T> for WavefunctionValue {
    fn act_on(&self, wf: &T, _cfg: &Array2<f64>) -> Result<OperatorValue, Error> {
        // need to square this, since "local value" is operator product / wave function value
        Ok(OperatorValue::Scalar(wf.current_value()?.0.powi(2)))
    }
}
