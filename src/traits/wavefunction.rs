use ndarray::{Array2, Array};

pub trait WaveFunction {

    type D;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Array<f64, Self::D>;

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> f64;
}

