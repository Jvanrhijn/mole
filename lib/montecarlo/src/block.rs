// Standard imports
use std::collections::HashMap;
use std::ops::{AddAssign, Sub, Div};
// Third party imports
use num_traits::{Float, identities::{Zero, One}};
use ndarray::{Axis, Array1};

/// Convenience struct for dealing with block averaging.
pub struct Block<T> {
    //values: Array2<T>,
    values: HashMap<String, Array1<T>>
}

impl<T> Block<T>
    where T: 'static + Clone + Copy + One + Zero + Float + AddAssign + Sub<Output=T> + Div<Output=T>
{
    pub fn new(size: usize, observables: &[&String]) -> Self {
        let values = observables.iter().map(|&name| (name.clone(), Array1::<T>::zeros(size)))
            .collect();
        Self{values}
    }

    pub fn set_value(&mut self, idx: usize, samples: &HashMap<String, T>) {
        for (key, sample) in samples.iter() {
           let value = self.values.get_mut(key)
               .expect("Observable not present in block HashMap");
            value[idx] = *sample;
        }
    }

    pub fn mean(&self) -> HashMap<String, T> {
        self.values.iter().map(|(key, array)| {
            let mean = array.mean_axis(Axis(0)).scalar_sum();
            (key.clone(), mean)
        }).collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block() {
        const SIZE: usize = 10;
        const MEAN: f64 = 4.5;
        let observables: &[&String] = &[&"Energy".to_string()];
        let mut data = HashMap::new();
        data.insert("Energy".to_string(), 0.0);
        let mut block = Block::<f64>::new(SIZE, observables);

        for i in 0..SIZE {
            *data.get_mut("Energy").unwrap() = i as f64;
            block.set_value(i, &data);
        }
        assert_eq!(block.mean()["Energy"], MEAN);
    }

}
