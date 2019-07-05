// Standard imports
use std::convert;
// Third party imports
use ndarray_linalg::error::LinalgError;
use std::fmt;

#[derive(Debug)]
pub enum Error {
    LinalgError(LinalgError),
    FuncError,
    OperatorValueAccessError,
    DataAccessError,
}

impl convert::From<LinalgError> for Error {
    fn from(e: LinalgError) -> Self {
        Error::LinalgError(e)
    }
}
