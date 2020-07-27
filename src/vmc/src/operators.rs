use errors::Error;
use ndarray::{Array2, Ix2};
use operator::{LocalOperator, OperatorValue};
use optimize::Optimize;
use wavefunction_traits::Function;

pub struct ParameterGradient;

impl<T: Optimize + Function<f64, D = Ix2>> LocalOperator<T> for ParameterGradient {
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue, Error> {
        Ok(OperatorValue::Vector(wf.parameter_gradient(cfg)?))
    }
}

#[derive(Copy, Clone)]
pub struct WavefunctionValue;

impl<T: Function<f64, D = Ix2>> LocalOperator<T> for WavefunctionValue {
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue, Error> {
        Ok(OperatorValue::Scalar(wf.value(cfg)?))
    }
}
