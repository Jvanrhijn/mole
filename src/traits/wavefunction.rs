use ndarray::Array;
use error::Error;

pub trait WaveFunction {

    type D;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Array<f64, Self::D>;

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error>;

    fn num_electrons(&self) -> usize;
}

