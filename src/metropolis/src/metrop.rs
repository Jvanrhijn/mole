use ndarray::{Array1, Array2, Ix2};
use ndarray_linalg::Norm;
use ndarray_rand::RandomExt;
use rand::distributions::{Normal, Range};
use rand::rngs::StdRng;
use rand::{FromEntropy, Rng};

use crate::traits::Metropolis;
use wavefunction::{Cache, Differentiate, Function};

#[allow(dead_code)]
type Vgl = (f64, Array2<f64>, f64);

/// Simplest Metropolis algorithm.
/// Transition matrix T(x -> x') is constant inside a cubical box,
/// and zero outside it. This yields an acceptance probability of
/// $A(x -> x') = \min(\psi(x')^2 / \psi(x)^2, 1)$.
pub struct MetropolisBox<R>
where
    R: Rng,
{
    box_side: f64,
    rng: R,
}

impl<R> MetropolisBox<R>
where
    R: Rng,
{
    pub fn from_rng(box_side: f64, rng: R) -> Self {
        Self { box_side, rng }
    }
}

impl MetropolisBox<StdRng> {
    pub fn new(box_side: f64) -> Self {
        Self {
            box_side,
            rng: StdRng::from_entropy(),
        }
    }
}

impl<T, R> Metropolis<T> for MetropolisBox<R>
where
    T: Differentiate + Function<f64, D = Ix2> + Cache<Array2<f64>, U = usize, V = Vgl>,
    R: Rng,
{
    type R = R;

    fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    fn propose_move(&mut self, wf: &mut T, cfg: &Array2<f64>, idx: usize) -> Array2<f64> {
        let mut config_proposed = cfg.clone();
        {
            let mut mov_slice = config_proposed.slice_mut(s![idx, ..]);
            mov_slice += &Array1::random_using(
                3,
                Range::new(-0.5 * self.box_side, 0.5 * self.box_side),
                &mut self.rng,
            );
        }
        wf.enqueue_update(idx, &config_proposed);
        config_proposed
    }

    fn accept_move(&mut self, wf: &mut T, _cfg: &Array2<f64>, _cfg_prop: &Array2<f64>) -> bool {
        let wf_value = wf
            .enqueued_value()
            .expect("Attempted to retrieve value from empty cache")
            .0;
        let acceptance = (wf_value.powi(2) / wf.current_value().0.powi(2)).min(1.0);
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

pub struct MetropolisDiffuse<R>
where
    R: Rng,
{
    time_step: f64,
    rng: R,
}

impl<R: Rng> MetropolisDiffuse<R> {
    pub fn from_rng(time_step: f64, rng: R) -> Self {
        Self { time_step, rng }
    }
}

impl MetropolisDiffuse<StdRng> {
    pub fn new(time_step: f64) -> Self {
        Self {
            time_step,
            rng: StdRng::from_entropy(),
        }
    }
}

impl<T, R> Metropolis<T> for MetropolisDiffuse<R>
where
    T: Differentiate + Function<f64, D = Ix2> + Cache<Array2<f64>, U = usize, V = Vgl>,
    R: Rng,
{
    type R = R;

    fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    fn propose_move(&mut self, wf: &mut T, cfg: &Array2<f64>, idx: usize) -> Array2<f64> {
        let mut config_proposed = cfg.clone();
        {
            let (wf_value, wf_grad, _) = wf.current_value();
            let drift_velocity = &wf_grad.slice(s![idx, ..]) / wf_value;

            let mut mov_slice = config_proposed.slice_mut(s![idx, ..]);
            mov_slice += &(drift_velocity * self.time_step);
            mov_slice +=
                &Array1::random_using(3, Normal::new(0.0, self.time_step.sqrt()), &mut self.rng);
        }
        wf.enqueue_update(idx, &config_proposed);
        config_proposed
    }

    fn accept_move(&mut self, wf: &mut T, cfg: &Array2<f64>, cfg_prop: &Array2<f64>) -> bool {
        let (wf_value, wf_grad, _) = wf
            .enqueued_value()
            .expect("Attempted to retrieve value from empty cache");
        let drift_velocity = &wf_grad / wf_value;

        let (wf_value_old, wf_grad_old, _) = wf.current_value();
        let drift_velocity_old = &wf_grad_old / wf_value_old;

        let cfg_difference = cfg - cfg_prop;
        let drift_velocity_difference = &drift_velocity - &drift_velocity_old;

        let exponent = (drift_velocity_old.norm_l2().powi(2) - drift_velocity.norm_l2().powi(2)
            + 2.0 * (&cfg_difference * &drift_velocity_difference).scalar_sum())
            / self.time_step;

        let acceptance =
            (exponent.exp() * wf_value.powi(2) / wf.current_value().0.powi(2)).min(1.0);
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
    use wavefunction::Error;

    // define stub wave function
    struct WaveFunctionMock {
        value: f64,
    }

    #[allow(dead_code)]
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

        fn gradient(&self, _cfg: &Array2<f64>) -> Result<Array2<f64>, Error> {
            unimplemented!()
        }

        fn laplacian(&self, _cfg: &Array2<f64>) -> Result<f64, Error> {
            Ok(1.0)
        }
    }

    impl Cache<Array2<f64>> for WaveFunctionMock {
        type A = Array2<f64>;
        type V = Vgl;
        type U = usize;
        fn refresh(&mut self, _new: &Array2<f64>) {}
        fn enqueue_update(&mut self, _ud: Self::U, _new: &Array2<f64>) {}
        fn push_update(&mut self) {}
        fn flush_update(&mut self) {}
        fn current_value(&self) -> Self::V {
            (self.value, Array2::ones((1, 1)) * self.value, self.value)
        }
        fn enqueued_value(&self) -> Option<Self::V> {
            Some((self.value, Array2::ones((1, 1)) * self.value, self.value))
        }
    }

    #[test]
    fn test_uniform_wf() {
        let cfg = Array2::<f64>::ones((1, 3));
        let mut wf = WaveFunctionMock { value: 1.0 };
        let mut metrop = MetropolisBox::<StdRng>::new(1.0);
        let new_cfg = metrop.propose_move(&mut wf, &cfg, 0); // should always accept
        assert!(metrop.accept_move(&mut wf, &cfg, &new_cfg));
    }

}
