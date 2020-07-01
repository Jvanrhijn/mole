use crate::traits::BranchingAlgorithm;
use ndarray::Array2;
use rand::RngCore;

use rand::distributions::{Distribution, Weighted, WeightedChoice};

pub struct SRBrancher;

impl SRBrancher {
    pub fn new() -> Self {
        Self {}
    }
}

impl<R: RngCore> BranchingAlgorithm<R> for SRBrancher {
    fn branch(
        &mut self,
        walkers: &Vec<(f64, Array2<f64>)>,
        rng: &mut R,
    ) -> Vec<(f64, Array2<f64>)> {
        let global_weight = walkers.iter().fold(0.0, |acc, (w, _)| acc + w) / walkers.len() as f64;
        let max_weight = walkers.iter().fold(0.0, |acc, (w, _)| f64::max(acc, *w));
        // normalize weights by the maximum weight
        let norm_factor = walkers.len() as f64 / max_weight;
        let mut confs_weighted: Vec<_> = walkers
            .iter()
            .map(|(w, c)| Weighted {
                weight: (w * norm_factor) as u32,
                item: c,
            })
            .collect();
        let wc = WeightedChoice::new(&mut confs_weighted);
        // construct new walkers
        let mut new_walkers = vec![];
        for _ in 0..walkers.len() {
            new_walkers.push((global_weight, wc.sample(rng).clone()));
        }
        new_walkers
    }
}

pub struct OptimalSRBrancher;

impl OptimalSRBrancher {
    pub fn new() -> Self {
        unimplemented!();
        Self {}
    }
}

impl<R: RngCore> BranchingAlgorithm<R> for OptimalSRBrancher {
    fn branch(
        &mut self,
        walkers: &Vec<(f64, Array2<f64>)>,
        rng: &mut R,
    ) -> Vec<(f64, Array2<f64>)> {
        let global_weight = walkers.iter().fold(0.0, |acc, (w, _)| acc + w) / walkers.len() as f64;
        let max_weight = walkers.iter().fold(0.0, |acc, (w, _)| f64::max(acc, *w));

        // obtain the positive and negative walkers
        let positive_walkers = walkers.iter().filter(|(w, _)| *w >= 1.0);
        let negative_walkers = walkers.iter().filter(|(w, _)| *w < 1.0);
        let num_reconf = positive_walkers.map(|(w, _)| (w/walkers.len() as f64 - 1.0).abs()).sum::<f64>();
        let num_reconf_minus = negative_walkers.map(|(w, _)| (w/walkers.len() as f64 - 1.0).abs()).sum::<f64>();


        // normalize weights by the maximum weight
        let norm_factor = walkers.len() as f64 / max_weight;
        let mut confs_weighted: Vec<_> = walkers
            .iter()
            .map(|(w, c)| Weighted {
                weight: (w * norm_factor) as u32,
                item: c,
            })
            .collect();
        let wc = WeightedChoice::new(&mut confs_weighted);

        // construct new walkers
        let mut new_walkers = vec![];
        for _ in 0..walkers.len() {
            new_walkers.push((global_weight, wc.sample(rng).clone()));
        }

        new_walkers
    }
}