use ndarray::{Array2, Ix2};
use traits::wavefunction::WaveFunction;
use traits::function::Function;

pub trait Metropolis<T: WaveFunction + Function<f64, D=Ix2>> {
    fn propose_move(&self, wf: &T, cfg: &Array2<f64>, idx: usize) -> Array2<f64>;

    fn accept_move(&self, wf: &T, cfg: &Array2<f64>, cfg_prop: &Array2<f64>) -> bool;

    fn move_state(&mut self, wf: &T, cfg: &Array2<f64>, idx: usize) -> Option<Array2<f64>>;

    fn wf_val_prev_mut(&mut self) -> &mut f64;
}