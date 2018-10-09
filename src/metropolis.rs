// Third party imports
use rand::{random};
use rand::distributions::Range;
use ndarray::{Array2, Ix2, Axis};
use ndarray_rand::RandomExt;
// First party imports
use traits::function::Function;

#[allow(dead_code)]
pub fn metropolis_single_move_box<T>(wf: &T, cfg: &Array2<f64>) -> Option<Array2<f64>>
where
    T: Function<f64, D=Ix2>
{
    let num_elecs = cfg.len_of(Axis(1));
    let electron_to_move = random::<usize>() % num_elecs;
    let mut mov = Array2::<f64>::zeros((cfg.len(), 3));
    {
        let mut mov_slice = mov.slice_mut(s![electron_to_move, ..]);
        mov_slice += &Array2::random((num_elecs, 3), Range::new(-1., 1.));
    }
    let cfg_proposed = cfg + &mov;
    let threshold = random::<f64>();
    let acceptance = (wf.value(&cfg_proposed).powi(2)/wf.value(cfg).powi(2)).min(1.);
    if acceptance > threshold {
        Some(cfg_proposed)
    } else {
        None
    }
}
