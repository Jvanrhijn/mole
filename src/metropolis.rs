// Third party imports
use rand::{random};
use rand::distributions::Range;
use ndarray::{Array1, Array2, Ix2, Axis};
use ndarray_rand::RandomExt;
// First party imports
use traits::function::Function;

#[allow(dead_code)]
pub fn metropolis_single_move_box<T>(wf: &T, cfg: &Array2<f64>, idx: usize) -> Option<Array2<f64>>
where
    T: Function<f64, D=Ix2>
{
    let num_elecs = cfg.len_of(Axis(0));
    let mut mov = Array2::<f64>::zeros((num_elecs, 3));
    {
        let mut mov_slice = mov.slice_mut(s![idx, ..]);
        mov_slice += &Array1::random(3, Range::new(-1., 1.));
    }
    let cfg_proposed = cfg + &mov;
    let threshold = random::<f64>();
    let acceptance = (wf.value(&cfg_proposed).unwrap().powi(2)/wf.value(cfg).unwrap().powi(2)).min(1.);
    if acceptance > threshold {
        Some(cfg_proposed)
    } else {
        None
    }
}
