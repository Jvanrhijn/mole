use std::vec::Vec;

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
}