use rand::random;
use rand::distributions::Range;
use ndarray::{Array1, Array2, Ix2};
use ndarray_rand::RandomExt;

use traits::metropolis::Metropolis;
use traits::function::Function;
use traits::wavefunction::WaveFunction;

#[allow(dead_code)]
pub struct MetropolisBox {
    box_side: f64,
    wf_value_prev: f64
}

impl MetropolisBox {
    pub fn new(box_side: f64) -> Self {
        Self{box_side, wf_value_prev: 0.}
    }

    pub fn set_wave_function_value(&mut self, wf_value: f64) {
        self.wf_value_prev = wf_value;
    }
}

impl<T: WaveFunction + Function<f64, D=Ix2>> Metropolis<T> for MetropolisBox {

    fn propose_move(&self, _wf: &T, cfg: &Array2<f64>, idx: usize) -> Array2<f64> {
        let mut config_proposed = cfg.clone();
        {
            let mut mov_slice = config_proposed.slice_mut(s![idx, ..]);
            mov_slice += &Array1::random(3, Range::new(-0.5*self.box_side, 0.5*self.box_side));
        }
        config_proposed
    }

    fn accept_move(&self, wf: &T, _cfg: &Array2<f64>, cfg_prop: &Array2<f64>) -> bool {
        let wf_value = wf.value(cfg_prop)
            .expect("Failed to evaluate wave function in moved configuration");
        let acceptance = (wf_value.powi(2)/self.wf_value_prev.powi(2)).min(1.);
        acceptance > random::<f64>()
    }

    fn move_state(&mut self, wf: &T, cfg: &Array2<f64>, idx: usize) -> Option<Array2<f64>> {
        let cfg_proposed = self.propose_move(wf, cfg, idx);
        if self.accept_move(wf, cfg, &cfg_proposed) {
            self.wf_value_prev = wf.value(&cfg_proposed)
                .expect("Failed to evaluate wave function at new configuration");
            Some(cfg_proposed)
        } else {
            None
        }
    }

    fn wf_val_prev_mut(&mut self) -> &mut f64 {
        &mut self.wf_value_prev
    }

}