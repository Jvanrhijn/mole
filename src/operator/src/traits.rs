// std imports
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};
// Third party imports
use errors::Error::{self, OperatorValueAccessError};
use ndarray::{Array1, Array2};

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, PartialEq, Clone)]
pub enum OperatorValue {
    Scalar(f64),
    Vector(Array1<f64>),
    Matrix(Array2<f64>),
}

impl fmt::Display for OperatorValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OperatorValue::Scalar(value) => write!(f, "{}", value),
            OperatorValue::Vector(value) => {
                let mut output = String::new();
                for x in value {
                    output = format!("{} {}", output, x);
                }
                write!(f, "{}", output)
            }
            _ => panic!("Display not implemented for Matrix values"),
        }
    }
}

impl OperatorValue {
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        match self {
            OperatorValue::Scalar(x) => OperatorValue::Scalar(f(*x)),
            OperatorValue::Vector(v) => OperatorValue::Vector(v.mapv(f)),
            OperatorValue::Matrix(m) => OperatorValue::Matrix(m.mapv(f)),
        }
    }

    pub fn get_scalar(&self) -> Result<&f64> {
        match self {
            OperatorValue::Scalar(value) => Ok(value),
            _ => Err(OperatorValueAccessError),
        }
    }

    pub fn get_vector(&self) -> Result<&Array1<f64>> {
        match self {
            OperatorValue::Vector(value) => Ok(value),
            _ => Err(OperatorValueAccessError),
        }
    }

    pub fn get_matrix(&self) -> Result<&Array2<f64>> {
        match self {
            OperatorValue::Matrix(value) => Ok(value),
            _ => Err(OperatorValueAccessError),
        }
    }
}

impl Add for OperatorValue {
    type Output = OperatorValue;

    fn add(self, other: OperatorValue) -> OperatorValue {
        use OperatorValue::*;
        match self {
            Scalar(value) => match other {
                Scalar(value_other) => Scalar(value + value_other),
                Vector(value_other) => Vector(value + value_other),
                Matrix(value_other) => Matrix(value + value_other),
            },
            Vector(value) => match other {
                Scalar(value_other) => Vector(value_other + value),
                Vector(value_other) => Vector(value + value_other),
                _ => unimplemented!(),
            },
            Matrix(value) => match other {
                Scalar(value_other) => Matrix(value_other + value),
                Matrix(value_other) => Matrix(value_other + value),
                _ => unimplemented!(),
            },
        }
    }
}

impl Sub for OperatorValue {
    type Output = OperatorValue;

    fn sub(self, other: OperatorValue) -> OperatorValue {
        use OperatorValue::*;
        match self {
            Scalar(value) => match other {
                Scalar(value_other) => Scalar(value - value_other),
                Vector(value_other) => Vector(value - value_other),
                Matrix(value_other) => Matrix(value - value_other),
            },
            Vector(value) => match other {
                Scalar(value_other) => Vector(value_other - value),
                Vector(value_other) => Vector(value_other - value),
                _ => unimplemented!(),
            },
            Matrix(value) => match other {
                Scalar(value_other) => Matrix(value_other - value),
                _ => unimplemented!(),
            },
        }
    }
}

impl Mul for OperatorValue {
    type Output = OperatorValue;

    fn mul(self, other: OperatorValue) -> OperatorValue {
        use OperatorValue::*;
        match self {
            Scalar(value) => match other {
                Scalar(value_other) => Scalar(value * value_other),
                Vector(value_other) => Vector(value * value_other),
                Matrix(value_other) => Matrix(value * value_other),
            },
            Vector(value) => match other {
                Scalar(value_other) => Vector(value_other * value),
                Vector(value_other) => Vector(value_other * value),
                _ => unimplemented!(),
            },
            Matrix(value) => match other {
                Scalar(value_other) => Matrix(value_other * value),
                _ => unimplemented!(),
            },
        }
    }
}

impl Div for OperatorValue {
    type Output = OperatorValue;

    fn div(self, other: OperatorValue) -> OperatorValue {
        use OperatorValue::*;
        match self {
            Scalar(value) => match other {
                Scalar(value_other) => Scalar(value / value_other),
                Vector(value_other) => Vector(value / value_other),
                Matrix(value_other) => Matrix(value / value_other),
            },
            Vector(value) => match other {
                Scalar(value_other) => Vector(value_other / value),
                _ => unimplemented!(),
            },
            Matrix(value) => match other {
                Scalar(value_other) => Matrix(value_other / value),
                _ => unimplemented!(),
            },
        }
    }
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
                Vector(value_other) => Vector(value_other + value),
                _ => unimplemented!(),
            },
            Matrix(value) => match other {
                Scalar(value_other) => Matrix(*value_other + value),
                _ => unimplemented!(),
            },
        }
    }
}

impl Sub for &OperatorValue {
    type Output = OperatorValue;

    fn sub(self, other: &OperatorValue) -> OperatorValue {
        use OperatorValue::*;
        match self {
            Scalar(value) => match other {
                Scalar(value_other) => Scalar(value - value_other),
                Vector(value_other) => Vector(*value - value_other),
                Matrix(value_other) => Matrix(*value - value_other),
            },
            Vector(value) => match other {
                Scalar(value_other) => Vector(*value_other - value),
                Vector(value_other) => Vector(value_other - value),
                _ => unimplemented!(),
            },
            Matrix(value) => match other {
                Scalar(value_other) => Matrix(*value_other - value),
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
                Vector(value_other) => Vector(value_other * value),
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

impl Sum for OperatorValue {
    fn sum<I: Iterator<Item = OperatorValue>>(iter: I) -> OperatorValue {
        iter.fold(OperatorValue::Scalar(0.0), |a, b| &a + &b)
    }
}

/// Interface for creating quantum operators that act on Function types.
pub trait Operator<T>: Send + Sync {
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue>;
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
