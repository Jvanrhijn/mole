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

    /// Calculate updated value of the cache given update data and new configuration
    fn update(&self, ud: Self::U, new: &T) -> (Vec<Self::A>, Self::V);

    /// Set the cache data directly
    fn set_cache(&mut self, mut storage: Vec<Self::A>, value: Self::V);

    /// Update the cache in place, overwriting old wave function data
    fn update_inplace(&mut self, ud: Self::U, new: &T);

    /// Return the current value of the cached data
    fn current_value(&self) -> Self::V;
}