use ndarray::Array2;
use rand::RngCore;

pub trait BranchingAlgorithm<R: RngCore> {
    fn branch(&mut self, walkers: &Vec<(f64, Array2<f64>)>, rng: &mut R) -> Vec<(f64, Array2<f64>)>;
}