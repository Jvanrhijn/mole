use std::vec::Vec;

/// convenience struct for dealing with block averaging in a simple way
pub struct Block {
    values: Vec<f64>,
    size: usize,
    idx: usize
}

impl Block {
    pub fn new(size: usize) -> Self {
        let mut values = Vec::<f64>::new();
        values.resize(size, 0.0);
        Self{values, size, idx: 0}
    }

    pub fn value_mut(&mut self, idx: usize) -> &mut f64 {
        &mut self.values[idx]
    }

    pub fn mean_and_variance(&self) -> (f64, f64) {
        let mean = self.values.iter().sum::<f64>()/self.size as f64;
        let variance = self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()/(self.size - 1) as f64;
        (mean, variance)
    }

    pub fn size(&self) -> usize {
        self.size
    }
}
