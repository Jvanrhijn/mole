// Standard imports
use std::collections::HashMap;
//Third party imports
use ndarray::{Array2, Ix2};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
use rand::Rng;
// First party imports
use crate::traits::*;
use metropolis::Metropolis;
use operator::{
    Operator,
    OperatorValue::{self, *},
};
use wavefunction_traits::{Cache, Differentiate, Function, WaveFunction};
use errors::Error;

/// Simple Monte Carlo sampler
/// Performs Metropolis step and keeps list of observables to sample
#[derive(Clone)]
pub struct Sampler<'a, T, V>
where
    T: Function<f64, D = Ix2> + Differentiate + Cache + Clone,
    V: Metropolis<T>,
{
    wave_function: T,
    config: Array2<f64>,
    metropolis: V,
    observables: &'a HashMap<String, Box<dyn Operator<T>>>,
    samples: HashMap<String, Vec<OperatorValue>>,
    acceptance: f64,
}

impl<'a, T, V> Sampler<'a, T, V>
where
    T: Function<f64, D = Ix2> + Differentiate + WaveFunction + Cache + Clone,
    V: Metropolis<T>,
    <V as Metropolis<T>>::R: Rng,
{
    pub fn new(
        mut wave_function: T,
        mut metrop: V,
        observables: &'a HashMap<String, Box<dyn Operator<T>>>,
    ) -> Self {
        let nelec = wave_function.num_electrons();
        let cfg = Array2::<f64>::random_using((nelec, 3), Range::new(-1., 1.), metrop.rng_mut());
        wave_function.refresh(&cfg);
        Self {
            wave_function,
            config: cfg,
            metropolis: metrop,
            observables,
            samples: HashMap::new(),
            acceptance: 0.0,
        }
    }

    pub fn with_initial_configuration(
        mut wave_function: T,
        metrop: V,
        observables: &'a HashMap<String, Box<dyn Operator<T>>>,
        cfg: Array2<f64>,
    ) -> Self {
        wave_function.refresh(&cfg);
        Self {
            wave_function,
            config: cfg,
            metropolis: metrop,
            observables: observables,
            samples: HashMap::new(),
            acceptance: 0.0,
        }
    }
}

impl<'a, T, V> MonteCarloSampler for Sampler<'a, T, V>
where
    T: Function<f64, D = Ix2> + Differentiate + WaveFunction + Cache + Clone,
    V: Metropolis<T, R: Rng>,
{
    type WaveFunc = T;

    fn sample(&mut self) -> Result<(), Error> {
        // First sample all observables on the current configuration
        let mut samples: HashMap<String, OperatorValue> = self
            .observables
            .iter()
            .map(|(name, operator)| {
                (
                    name.clone(),
                    &operator
                        .act_on(&self.wave_function, &self.config)
                        // TODO: find a way to get rid of this expect
                        .expect("Failed to act on wave function with operator")
                        / &Scalar(self.wave_function.current_value().0),
                )
            })
            .collect();
        // save the sampled values
        samples
            .drain()
            .for_each(|(name, value)| self.samples.entry(name).or_default().push(value));
        Ok(())
    }

    fn move_state(&mut self) {
        for e in 0..self.wave_function.num_electrons() {
            if let Some(config) =
                self.metropolis
                    .move_state(&mut self.wave_function, &self.config, e)
            {
                self.config = config;
                self.wave_function.push_update();
                self.acceptance += 1.0 / self.wave_function.num_electrons() as f64;
            } else {
                self.wave_function.flush_update();
            }
        }
        self.wave_function.refresh(&self.config);
    }

    fn data(&self) -> &HashMap<String, Vec<OperatorValue>> {
        &self.samples
    }

    fn num_observables(&self) -> usize {
        self.observables.len()
    }

    fn acceptance(&self) -> f64 {
        self.acceptance
    }

    fn observable_names(&self) -> Vec<&String> {
        self.observables.keys().collect()
    }

    fn consume_result(self) -> MonteCarloResult<Self::WaveFunc> {
        MonteCarloResult {
            wave_function: self.wave_function,
            data: self.samples,
        }
    }

    fn wave_function(&self) -> &Self::WaveFunc {
        &self.wave_function
    }

    fn wave_function_mut(&mut self) -> &mut Self::WaveFunc {
        &mut self.wave_function
    }

    fn reseed_rng(&mut self, s: [u8; 32]) {
        self.metropolis.reseed_rng(s);
    }

    fn generate_seed(&mut self) -> [u8; 32] {
        self.metropolis.rng_mut().gen::<[u8; 32]>()
    }
}
