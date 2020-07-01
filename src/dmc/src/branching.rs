use ndarray::Array2;
use crate::traits::BranchingAlgorithm;
use rand::RngCore;

use rand::distributions::{Distribution, Weighted, WeightedChoice};

pub struct SRBrancher;

impl SRBrancher {
    pub fn new() -> Self {
        Self {}
    }
}

impl<R: RngCore> BranchingAlgorithm<R> for SRBrancher {
    fn branch(&mut self, walkers: &Vec<(f64, Array2<f64>)>, rng: &mut R) -> Vec<(f64, Array2<f64>)> {
        let global_weight = walkers.iter().fold(0.0, |acc, (w, _)| acc + w);
        let new_weight = global_weight / walkers.len() as f64;
        let max_weight = walkers.iter().fold(0.0, |acc, (w, _)| f64::max(acc, *w));
        // normalize weights by the maximum weight
        let norm_factor = walkers.len() as f64 / max_weight;
        let mut confs_weighted: Vec<_> = walkers
            .iter()
            .map(|(w, c)| Weighted {
                weight: (w*norm_factor) as u32,
                item: c,
            })
            .collect();
        let wc = WeightedChoice::new(&mut confs_weighted);
        // construct new walkers
        let mut new_walkers = vec![];
        for _ in 0..walkers.len() {
            new_walkers.push((new_weight, wc.sample(rng).clone()));
        }
        new_walkers
    }
}