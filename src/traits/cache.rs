pub trait Cache<T> {
    /// storage type e.g. ndarray::Array2
    type A;
    /// value type e.g. (f64, Array2<f64>, f64) (value/grad/laplacian)
    type V;
    /// update data e.g. usize moved-electron-index
    type U;

    /// Refresh the cached data
    fn refresh(&mut self, new: T);

    /// Update the cache with a new T and any data needed to update the cache
    fn update(&mut self, ud: Self::U, new: T);

    /// Return the current value of the cached data
    fn current_value(&self) -> Self::V;
}