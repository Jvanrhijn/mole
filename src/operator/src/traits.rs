// std imports
use std::ops::{Add, Div, Mul};
// Third party imports
use ndarray::{Array1, Array2};
use wavefunction::Error;

#[derive(Debug, PartialEq)]
pub enum OperatorValue {
    Scalar(f64),
    Vector(Array1<f64>),
    Matrix(Array2<f64>),
}

impl Add for &OperatorValue {
    type Output = OperatorValue;

    fn add(self, other: &OperatorValue) -> OperatorValue {
        use OperatorValue::*;
        match self {
            Scalar(value) => match other {
                Scalar(value_other) => Scalar(value + value_other),
                Vector(value_other) => Vector(*value + value_other),
                Matrix(value_other) => Matrix(*value + value_other),
            },
            Vector(value) => match other {
                Scalar(value_other) => Vector(*value_other + value),
                _ => unimplemented!(),
            },
            Matrix(value) => match other {
                Scalar(value_other) => Matrix(*value_other + value),
                _ => unimplemented!(),
            },
        }
    }
}

impl Mul for &OperatorValue {
    type Output = OperatorValue;

    fn mul(self, other: &OperatorValue) -> OperatorValue {
        use OperatorValue::*;
        match self {
            Scalar(value) => match other {
                Scalar(value_other) => Scalar(value * value_other),
                Vector(value_other) => Vector(*value * value_other),
                Matrix(value_other) => Matrix(*value * value_other),
            },
            Vector(value) => match other {
                Scalar(value_other) => Vector(*value_other * value),
                _ => unimplemented!(),
            },
            Matrix(value) => match other {
                Scalar(value_other) => Matrix(*value_other * value),
                _ => unimplemented!(),
            },
        }
    }
}

impl Div for &OperatorValue {
    type Output = OperatorValue;

    fn div(self, other: &OperatorValue) -> OperatorValue {
        use OperatorValue::*;
        match self {
            Scalar(value) => match other {
                Scalar(value_other) => Scalar(value / value_other),
                Vector(value_other) => Vector(*value / value_other),
                Matrix(value_other) => Matrix(*value / value_other),
            },
            Vector(value) => match other {
                Scalar(value_other) => Vector(*value_other / value),
                _ => unimplemented!(),
            },
            Matrix(value) => match other {
                Scalar(value_other) => Matrix(*value_other / value),
                _ => unimplemented!(),
            },
        }
    }
}

/// Interface for creating quantum operators that act on Function types.
pub trait Operator<T> {
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue, Error>;
}



#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{num, prelude::*};

    proptest! {
        #[test]
        fn add_op_values(x in num::f64::NORMAL, y in num::f64::NORMAL) {
            let first = OperatorValue::Scalar(x);
            let second = OperatorValue::Scalar(y);
            prop_assert_eq!(
                x + y,
                match &first + &second{
                    OperatorValue::Scalar(value) => value,
                    _ => unimplemented!(),
                }
            );
        }
    }

}
