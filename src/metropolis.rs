use std::cmp;

use rand::{Range, random};
use ndarray::{Array1};
use ndarray_rand;

pub fn metropolis_single_move_box<T: WaveFunction>(wf: &T, cfg: &Array1<f64>) -> Option(Array1<f64>) {
    let electron_to_move = random::<usize>() % cfg.size() / 3;
    let mut mov = Array1::zeros(cfg.size());
    mov[[3*electron_to_move..3*electron_to_move+3]] += Array1::random((1, 3), Range::new(-1., 1.));
    let cfg_proposed = cfg + mov;
    let threshold = random::<f64>();
    let acceptance_prob = cmp::min(1., (wf.value(cfg_proposed)/wf.value(cfg)).abs());
    if acceptance_prob > threshold {
        Some(cfg_proposed)
    } else {
        None
    }
}