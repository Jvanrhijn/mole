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

    pub fn mean(&self) -> f64 {
        self.values.iter().fold(0.0, |acc, x| acc + x)/self.size as f64
    }

    pub fn size(&self) -> usize {
        self.size
    }
}
