// Standard imports
use std::convert;
// Third party imports
use ndarray_linalg::error::LinalgError;

#[derive(Debug)]
pub enum Error {
    LinalgError,
    FuncError,
}

impl convert::From<LinalgError> for Error {
    fn from(_e: LinalgError) -> Self {
        Error::LinalgError
    }
}
