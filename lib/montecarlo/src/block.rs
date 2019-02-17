// Standard imports
use std::vec::Vec;
use std::ops::{AddAssign, Sub, Div};
// Third party imports
use num_traits::{Float, identities::{Zero, One}};
use ndarray::{Axis, Array1, Array2};

/// Convenience struct for dealing with block averaging.
pub struct Block<T> {
    values: Array2<T>,
}

impl<T> Block<T>
    where T: 'static + Clone + Copy + One + Zero + Float + AddAssign + Sub<Output=T> + Div<Output=T>
{
    pub fn new(size: usize, num_observables: usize) -> Self {
        Self{values: Array2::<T>::zeros((size, num_observables))}
    }

    pub fn set_value(&mut self, idx: usize, value: Vec<T>) {
        let mut slice = self.values.slice_mut(s![idx, ..]);
        slice += &Array1::<T>::from_vec(value);
    }

    pub fn mean(&self) -> Array1<T> {
        self.values.mean_axis(Axis(0))
    }

    #[allow(dead_code)]
    pub fn variance(&self) -> Array1<T> {
        self.values.var_axis(Axis(0), T::zero())
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block() {
        const SIZE: usize = 10;
        const MEAN: f64 = 4.5;
        const VARIANCE: f64 = 8.25;
        let mut block = Block::<f64>::new(SIZE, 1);
        for i in 0..SIZE {
            block.set_value(i, vec![i as f64]);
        }
        assert_eq!(block.mean()[0], MEAN);
        assert_eq!(block.variance()[0], VARIANCE);
    }

}
