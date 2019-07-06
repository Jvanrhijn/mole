use errors::Error;
use ndarray::{Array, Array2};

type Vgl = (f64, Array2<f64>, f64);
type Ovgl = (Option<f64>, Option<Array2<f64>>, Option<f64>);

type Result<T> = std::result::Result<T, Error>;

pub trait Cache {
    /// update data e.g. usize moved-electron-index
    type U;

    /// Refresh the cached data
    fn refresh(&mut self, new: &Array2<f64>) -> Result<()>;

    /// Calculate updated value of the cache given update data and new configuration,
    /// and set this data enqueued
    fn enqueue_update(&mut self, ud: Self::U, new: &Array2<f64>) -> Result<()>;

    /// Push enqueued update into cache
    fn push_update(&mut self);

    /// Flush the enqueued update data
    fn flush_update(&mut self);

    /// Return the current value of the cached data
    fn current_value(&self) -> Result<Vgl>;

    /// Return enqueued value
    fn enqueued_value(&self) -> Ovgl;
}

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
}