use ndarray::Array;
use crate::error::Error;

pub trait Cache<T> {
    /// storage type e.g. ndarray::Array2
    type A;
    /// value type e.g. (f64, Array2<f64>, f64) (value/grad/laplacian)
    type V;
    /// update data e.g. usize moved-electron-index
    type U;

    /// Refresh the cached data
    fn refresh(&mut self, new: &T);

    /// Calculate updated value of the cache given update data and new configuration,
    /// and set this data enqueued
    fn enqueue_update(&mut self, ud: Self::U, new: &T);

    /// Push enqueued update into cache
    fn push_update(&mut self);

    /// Flush the enqueued update data
    fn flush_update(&mut self);

    /// Return the current value of the cached data
    fn current_value(&self) -> Self::V;

    /// Return enqueued value
    fn enqueued_value(&self) -> Option<Self::V>;
}

/// Interface for dealing with functions f: F^n -> F, where F is any field.
pub trait Function<T> {

    type D;

    fn value(&self, cfg: &Array<T, Self::D>) -> Result<T, Error>;
}

/// Interface for creating once- and twice differentiable functions.
pub trait Differentiate {

    type D;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Array<f64, Self::D>;

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error>;

}

pub trait WaveFunction {
    fn num_electrons(&self) -> usize;
}


