use crate::traits::BranchingAlgorithm;
use ndarray::Array2;
use rand::{RngCore, Rng};

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

        // number of reconfigurations to be done
        let num_reconf = (positive_walkers.clone().map(|(w, _)| (w/walkers.len() as f64).abs()).sum::<f64>() + rng.gen::<f64>()) as usize;


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

pub struct SimpleBranching;

impl SimpleBranching {
    pub fn new() -> Self {
        Self {}
    }
}

impl<R: RngCore + Rng> BranchingAlgorithm<R> for SimpleBranching {
    fn branch(
        &mut self,
        walkers: &Vec<(f64, Array2<f64>)>,
        rng: &mut R,
    ) -> Vec<(f64, Array2<f64>)> {
        let mut to_kill = vec![];
        let mut to_birth = vec![];
        let mut new_walkers = walkers.clone();
        for (i, (w, conf)) in walkers.iter().enumerate() {
            let num_copies = ((w + rng.gen::<f64>()) as usize).min(3);
            if num_copies == 0 {
                to_kill.push(i);
            } else {
                for _ in 0..num_copies-1 {
                    to_birth.push((*w, conf.clone()));
                }
            }
        }
        for idx in to_kill.iter().rev() {
            new_walkers.remove(*idx);
        }
        new_walkers.extend(to_birth);
        let num_new_walkers = new_walkers.len();
        let num_walkers = walkers.len();
        if num_walkers > num_new_walkers {
            let mut to_clone = vec![];
            // randomly clone some walkers
            let excess = num_walkers - num_new_walkers;
            for _ in 0..excess {
                to_clone.push(walkers[rng.gen_range::<usize>(0, num_walkers)].clone());
            } 
            new_walkers.extend(to_clone);
        } else {
            // randomly kill some walkers
            let excess = num_new_walkers - num_walkers;
            for _ in 0..excess {
                new_walkers.remove(rng.gen_range::<usize>(0, new_walkers.len()));
            }
        }
        new_walkers
    }
}