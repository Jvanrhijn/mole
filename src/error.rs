use std::error;
use std::fmt;
use std::convert;

use ndarray_linalg::error::LinalgError;

#[derive(Clone, Debug)]
pub struct FuncError;

impl fmt::Display for FuncError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "failed to evaluate function")
    }
}

impl error::Error for FuncError {
    fn description(&self) -> &str {
        "failed to evaluate function"
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

#[derive(Debug)]
pub enum Error {
    LinalgError,
    FuncError
}

impl convert::From<LinalgError> for Error {
    fn from(e: LinalgError) -> Self {
        Error::LinalgError
    }
}
