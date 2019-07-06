use ndarray::{Array1, Array2, Ix2};
use ndarray_linalg::Norm;
use ndarray_rand::RandomExt;
use rand::distributions::{Normal, Range};
use rand::rngs::StdRng;
use rand::{FromEntropy, Rng, SeedableRng};

use errors::Error;
use crate::traits::Metropolis;
use wavefunction_traits::{Cache, Differentiate, Function};

type Result<T> = std::result::Result<T, Error>;

#[allow(dead_code)]
type Vgl = (f64, Array2<f64>, f64);

/// Simplest Metropolis algorithm.
/// Transition matrix T(x -> x') is constant inside a cubical box,
/// and zero outside it. This yields an acceptance probability of
/// $A(x -> x') = \min(\psi(x')^2 / \psi(x)^2, 1)$.
#[derive(Clone)]
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
    T: Differentiate + Function<f64, D = Ix2> + Cache<U = usize> + Clone,
    R: Rng + SeedableRng,
    <R as SeedableRng>::Seed: From<[u8; 32]>,
{
    type R = R;

    fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    fn propose_move(&mut self, wf: &mut T, cfg: &Array2<f64>, idx: usize) -> Result<Array2<f64>> {
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
        Ok(config_proposed)
    }

    fn accept_move(&mut self, wf: &mut T, _cfg: &Array2<f64>, _cfg_prop: &Array2<f64>) -> Result<bool> {
        let wf_value = match wf.enqueued_value() {
            (Some(v), _, _) => v,
            _ => wf.current_value()?.0,
        };
        let acceptance = (wf_value.powi(2) / wf.current_value()?.0.powi(2)).min(1.0);
        Ok(acceptance > self.rng.gen::<f64>())
    }

    fn move_state(&mut self, wf: &mut T, cfg: &Array2<f64>, idx: usize) -> Result<Option<Array2<f64>>> {
        let cfg_proposed = self.propose_move(wf, cfg, idx)?;
        if self.accept_move(wf, cfg, &cfg_proposed)? {
            Ok(Some(cfg_proposed))
        } else {
            Ok(None)
        }
    }

    fn reseed_rng(&mut self, s: [u8; 32]) {
        self.rng = Self::R::from_seed(s.into());
    }
}

#[derive(Clone)]
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
    T: Differentiate + Function<f64, D = Ix2> + Cache<U = usize> + Clone,
    R: Rng + SeedableRng,
    <R as SeedableRng>::Seed: From<[u8; 32]>,
{
    type R = R;

    fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    fn propose_move(&mut self, wf: &mut T, cfg: &Array2<f64>, idx: usize) -> Result<Array2<f64>> {
        let mut config_proposed = cfg.clone();
        {
            let (wf_value, wf_grad, _) = wf.current_value()?;
            let drift_velocity = &wf_grad.slice(s![idx, ..]) / wf_value;

            let mut mov_slice = config_proposed.slice_mut(s![idx, ..]);
            mov_slice += &(drift_velocity * self.time_step);
            mov_slice +=
                &Array1::random_using(3, Normal::new(0.0, self.time_step.sqrt()), &mut self.rng);
        }
        wf.enqueue_update(idx, &config_proposed);
        Ok(config_proposed)
    }

    fn accept_move(&mut self, wf: &mut T, cfg: &Array2<f64>, cfg_prop: &Array2<f64>) -> Result<bool> {
        let (wf_value, wf_grad) = match wf.enqueued_value() {
            (Some(v), Some(g), _) => (v, g),
            _ => (wf.current_value()?.0, wf.current_value()?.1),
        };
        let drift_velocity = &wf_grad / wf_value;
        let (wf_value_old, wf_grad_old, _) = wf.current_value()?;
        let drift_velocity_old = &wf_grad_old / wf_value_old;

        let exponent = -1.0 / (2.0 * self.time_step)
            * (2.0
                * ((&drift_velocity + &drift_velocity_old) * (cfg_prop - cfg)).scalar_sum()
                * self.time_step
                + self.time_step.powi(2)
                    * (drift_velocity.norm_l2() - drift_velocity_old.norm_l2()));

        let acceptance =
            (exponent.exp() * wf_value.powi(2) / wf.current_value()?.0.powi(2)).min(1.0);

        Ok(acceptance > self.rng.gen::<f64>())
    }

    fn move_state(&mut self, wf: &mut T, cfg: &Array2<f64>, idx: usize) -> Result<Option<Array2<f64>>> {
        let cfg_proposed = self.propose_move(wf, cfg, idx)?;
        if self.accept_move(wf, cfg, &cfg_proposed)? {
            Ok(Some(cfg_proposed))
        } else {
            Ok(None)
        }
    }

    fn reseed_rng(&mut self, s: [u8; 32]) {
        self.rng = Self::R::from_seed(s.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use errors::Error;

    // define stub wave function
    #[derive(Clone)]
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

        fn value(&self, _cfg: &Array2<f64>) -> Result<f64> {
            Ok(self.value)
        }
    }

    impl Differentiate for WaveFunctionMock {
        type D = Ix2;

        fn gradient(&self, _cfg: &Array2<f64>) -> Result<Array2<f64>> {
            unimplemented!()
        }

        fn laplacian(&self, _cfg: &Array2<f64>) -> Result<f64> {
            Ok(1.0)
        }
    }

    type Ovgl = (Option<f64>, Option<Array2<f64>>, Option<f64>);

    impl Cache for WaveFunctionMock {
        type U = usize;
        fn refresh(&mut self, _new: &Array2<f64>) -> Result<()> { Ok(()) }
        fn enqueue_update(&mut self, _ud: Self::U, _new: &Array2<f64>) -> Result<()> { Ok(()) }
        fn push_update(&mut self) {}
        fn flush_update(&mut self) {}
        fn current_value(&self) -> Result<Vgl> {
            Ok((self.value, Array2::ones((1, 1)) * self.value, self.value))
        }
        fn enqueued_value(&self) -> Ovgl {
            (
                Some(self.value),
                Some(Array2::ones((1, 1)) * self.value),
                Some(self.value),
            )
        }
    }

    #[test]
    fn test_uniform_wf() {
        let cfg = Array2::<f64>::ones((1, 3));
        let mut wf = WaveFunctionMock { value: 1.0 };
        let mut metrop = MetropolisBox::<StdRng>::new(1.0);
        let new_cfg = metrop.propose_move(&mut wf, &cfg, 0); // should always accept
        assert!(metrop.accept_move(&mut wf, &cfg, &new_cfg).unwrap());
    }

}
