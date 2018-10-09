use ndarray::{Array};

pub trait Function<T> {
    type D;
    fn value(&self, cfg: &Array<T, Self::D>) -> T;
}
