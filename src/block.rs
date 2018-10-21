// Standard imports
use std::vec::Vec;
// Third party imports
use ndarray::{Axis, Array1, Array2};

/// Convenience struct for dealing with block averaging.
pub struct Block {
    values: Array2<f64>,
    size: usize,
    idx: usize
}

impl Block {
    pub fn new(size: usize, num_observables: usize) -> Self {
        Self{values: Array2::<f64>::zeros((size, num_observables)), size, idx: 0}
    }

    pub fn set_value(&mut self, idx: usize, value: Vec<f64>) {
        let mut slice = self.values.slice_mut(s![idx, ..]);
        slice += &Array1::<f64>::from_vec(value);
    }

    pub fn mean(&self) -> Array1<f64> {
        self.values.mean_axis(Axis(0))
    }

     pub fn variance(&self) -> Array1<f64> {
         self.values.var_axis(Axis(0), 0.0)
     }

    pub fn size(&self) -> usize {
        self.size
    }
}
