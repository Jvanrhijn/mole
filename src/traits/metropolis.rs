use ndarray::{Array1, Array2, Ix2};
use traits::wavefunction::WaveFunction;
use traits::function::Function;

pub trait Metropolis {
    fn propose_move<T: WaveFunction + Function<f64, D=Ix2>>(&self, wf: &T, cfg: &Array2<f64>, idx: usize) -> Array2<f64>;

    fn accept_move<T: WaveFunction + Function<f64, D=Ix2>>(&self, wf: &T, cfg: &Array2<f64>, cfg_prop: &Array2<f64>) -> bool;

    fn move_state<T: WaveFunction + Function<f64, D=Ix2>>(&mut self, wf: &T, cfg: &Array2<f64>, idx: usize) -> Option<Array2<f64>>;
}