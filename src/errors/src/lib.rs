// Standard imports
use std::convert;
// Third party imports
use ndarray::ShapeError;
use ndarray_linalg::error::LinalgError;

#[derive(Debug)]
pub enum Error {
    LinalgError(LinalgError),
    ShapeError(ShapeError),
    FuncError,
    OperatorValueAccessError,
    DataAccessError,
    EmptyCacheError,
}

impl convert::From<LinalgError> for Error {
    fn from(e: LinalgError) -> Self {
        Error::LinalgError(e)
    }
}

impl convert::From<ShapeError> for Error {
    fn from(e: ShapeError) -> Self {
        Error::ShapeError(e)
    }
}
