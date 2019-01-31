use rand::{FromEntropy, Rng};
use rand::rngs::SmallRng;
use rand::distributions::Range;
use ndarray::{Array1, Array2, Ix2};
use ndarray_rand::RandomExt;

use traits::metropolis::Metropolis;
use traits::function::Function;
use traits::differentiate::Differentiate;
use traits::cache::Cache;

/// Simplest Metropolis algorithm.
/// Transition matrix T(x -> x') is constant inside a cubical box,
/// and zero outside it. This yields an acceptance probability of
/// $A(x -> x') = \min(\psi(x')^2 / \psi(x)^2, 1)$.
pub struct MetropolisBox<R> where R: Rng {
    box_side: f64,
    wf_value_prev: f64,
    rng: R
}

impl<R> MetropolisBox<R> where R: Rng {

    pub fn from_rng(box_side: f64, rng: R) -> Self {
        Self{box_side, wf_value_prev: 0.0, rng}
    }

}

impl MetropolisBox<SmallRng> {
    pub fn new(box_side: f64) -> Self {
        let rng = SmallRng::from_entropy();
        Self{box_side, wf_value_prev: 0.0, rng}
    }
}

impl<T, R> Metropolis<T> for MetropolisBox<R>
where T: Differentiate + Function<f64, D=Ix2> + Cache<Array2<f64>, U=usize, V=(f64, f64)>,
      R: Rng
{
    type R = R;

    fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    fn propose_move(&mut self, wf: &mut T, cfg: &Array2<f64>, idx: usize) -> Array2<f64> {
        let mut config_proposed = cfg.clone();
        {
            let mut mov_slice = config_proposed.slice_mut(s![idx, ..]);
            mov_slice += &Array1::random_using(3, Range::new(-0.5*self.box_side, 0.5*self.box_side), &mut self.rng);
        }
        wf.enqueue_update(idx, &config_proposed);
        config_proposed
    }

    fn accept_move(&mut self, wf: &mut T, _cfg: &Array2<f64>, _cfg_prop: &Array2<f64>) -> bool {
        let wf_value = wf.enqueued_value()
            .expect("Attempted to retrieve value from empty cache").0;
        let acceptance = (wf_value.powi(2)/wf.current_value().0.powi(2)).min(1.0);
        acceptance > self.rng.gen::<f64>()
    }

    fn move_state(&mut self, wf: &mut T, cfg: &Array2<f64>, idx: usize) -> Option<Array2<f64>> {
        let cfg_proposed = self.propose_move(wf, cfg, idx);
        if self.accept_move(wf, cfg, &cfg_proposed) {
            Some(cfg_proposed)
        } else {
            None
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use error::Error;

    // define stub wave function
    struct WaveFunctionMock {
        value: f64
    }

    impl WaveFunctionMock {
        pub fn set_value(&mut self, new_val: f64) {
            self.value = new_val;
        }
    }

    impl Function<f64> for WaveFunctionMock {
        type D = Ix2;

        fn value(&self, _cfg: &Array2<f64>) -> Result<f64, Error> {
            Ok(self.value)
        }
    }

    impl Differentiate for WaveFunctionMock {
        type D = Ix2;

        fn gradient(&self, cfg: &Array2<f64>) -> Array2<f64> {
            cfg.clone()
        }

        fn laplacian(&self, _cfg: &Array2<f64>) -> Result<f64, Error> {
            Ok(1.0)
        }
    }

    impl Cache<Array2<f64>> for WaveFunctionMock {
        type A = Array2<f64>;
        type V = (f64, f64);
        type U = usize;
        fn refresh(&mut self, _new: &Array2<f64>) {}
        fn enqueue_update(&mut self, _ud: Self::U, _new: &Array2<f64>) {}
        fn push_update(&mut self) {}
        fn flush_update(&mut self) {}
        fn current_value(&self) -> Self::V {
            (self.value, self.value)
        }
        fn enqueued_value(&self) -> Option<Self::V> {
            Some((self.value, self.value))
        }
    }

    #[test]
    fn test_uniform_wf() {
        let cfg = Array2::<f64>::ones((1, 3));
        let mut wf = WaveFunctionMock{value: 1.0};
        let mut metrop = MetropolisBox::<SmallRng>::new(1.0);
        let new_cfg = metrop.propose_move(&mut wf, &cfg, 0); // should always accept
        assert!(metrop.accept_move(&mut wf, &cfg, &new_cfg));
    }

}