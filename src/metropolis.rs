use rand::{random};
use rand::distributions::Range;
use ndarray::{Array1};
use ndarray_rand::RandomExt;

use traits::wavefunction::WaveFunction;

#[allow(dead_code)]
pub fn metropolis_single_move_box<T: WaveFunction>(wf: &T, cfg: &Array1<f64>) -> Option<Array1<f64>> {
    let electron_to_move = random::<usize>() % cfg.len() / 3;
    let mut mov = Array1::zeros(cfg.len());
    {
        let mut mov_slice = mov.slice_mut(s![3*electron_to_move..3*electron_to_move+3]);
        mov_slice += &Array1::random(3, Range::new(-1., 1.));
    }
    let cfg_proposed = cfg + &mov;
    let threshold = random::<f64>();
    let acceptance = (wf.value(&cfg_proposed).abs().powi(2)/wf.value(cfg).abs().powi(2)).min(1.);
    if acceptance > threshold {
        Some(cfg_proposed)
    } else {
        None
    }
}
