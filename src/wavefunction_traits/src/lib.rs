use errors::Error;
use ndarray::Array;

type Result<T> = std::result::Result<T, Error>;

/// Interface for dealing with functions f: F^n -> F, where F is any field.
pub trait Function<T> {
    type D;

    fn value(&self, cfg: &Array<T, Self::D>) -> Result<T>;
}

/// Interface for creating once- and twice differentiable functions.
pub trait Differentiate {
    type D;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Result<Array<f64, Self::D>>;

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64>;
}

pub trait WaveFunction {
    fn num_electrons(&self) -> usize;

    fn dimension(&self) -> usize;
}
