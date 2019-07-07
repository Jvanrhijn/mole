use std::collections::HashMap;

use montecarlo::traits::{Log, MonteCarloResult, MonteCarloSampler};
use montecarlo::Runner;
use operator::OperatorValue;
use optimize::{Optimize, Optimizer};
use wavefunction_traits::{Cache, Differentiate, Function, WaveFunction};

use ndarray::{Array1, Ix2};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use errors::Error::{self, DataAccessError};

struct EmptyLogger;
impl Log for EmptyLogger {
    fn log(&mut self, _data: &HashMap<String, Vec<OperatorValue>>) -> String {
        String::new()
    }
}

pub struct VmcRunner<S, L, O> {
    logger: L,
    optimizer: O,
    sampler: S,
}

impl<S, T, L, O> VmcRunner<S, L, O>
where
    O: Optimizer + Send + Sync + Clone,
    T: WaveFunction
        + Differentiate
        + Function<f64, D = Ix2>
        + Cache
        + Optimize
        + Clone
        + Send
        + Sync,
    L: Log + Clone + Send + Sync,
    S: MonteCarloSampler<WaveFunc = T> + Clone + Send + Sync,
{
    pub fn new(sampler: S, optimizer: O, logger: L) -> Self {
        Self {
            logger,
            optimizer,
            sampler,
        }
    }

    #[allow(dead_code)]
    pub fn run_optimization(
        mut self,
        iters: usize,
        total_samples: usize,
        block_size: usize,
        nworkers: usize,
    ) -> Result<(T, Array1<f64>, Array1<f64>), Error> {
        let steps = total_samples / nworkers;

        let mut energies = Vec::new();
        let mut energy_errs = Vec::new();

        for _ in 0..iters {
            let samplers = vec![self.sampler.clone(); nworkers];

            // produce rng seeds for threads
            let seeds: Vec<_> = (0..nworkers)
                .map(|_| self.sampler.generate_seed())
                .collect();

            let results: Result<Vec<_>, _> = samplers
                .into_par_iter()
                .enumerate()
                .zip(seeds.into_par_iter())
                .map(|((worker, mut sampler), seed)| {
                    sampler.reseed_rng(seed);

                    if worker == 0 {
                        Runner::new(sampler, self.logger.clone()).run(steps, block_size)
                    } else {
                        Runner::new(sampler, EmptyLogger).run(steps, block_size)
                    }
                })
                .collect();

            let (mc_data, acceptance) = Self::concatenate_worker_data(&results?);

            let (averages, errors) = Self::process_monte_carlo_results(&mc_data, block_size)?;

            energies.push(*averages["Energy"].get_scalar()?);
            energy_errs.push(*errors["Energy"].get_scalar()?);

            let deltap = self.optimizer.compute_parameter_update(
                self.sampler.wave_function().parameters(),
                &averages,
                &mc_data,
            )?;

            self.sampler.wave_function_mut().update_parameters(&deltap);

            println!(
                "Energy:      {:.8} +/- {:.9}    accept: {:.8}",
                energies.last().expect("No samples present"),
                energy_errs.last().expect("No samples present"),
                acceptance / (total_samples as f64),
            );
        }

        Ok((
            self.sampler.wave_function().clone(),
            Array1::from_vec(energies),
            Array1::from_vec(energy_errs),
        ))
    }

    fn concatenate_worker_data(
        worker_data: &Vec<MonteCarloResult<T>>,
    ) -> (HashMap<String, Vec<OperatorValue>>, f64) {
        let mut full_data: HashMap<String, Vec<OperatorValue>> = HashMap::new();
        // concatenate all MC worker results
        let mut accept = 0.0;
        worker_data
            .iter()
            .for_each(|MonteCarloResult { wave_function: _, acceptance, data }| {
                accept += acceptance;
                data.iter().for_each(|(key, data)| {
                    full_data
                        .entry(key.to_string())
                        .or_insert_with(|| Vec::new())
                        .append(&mut (data.clone()));
                })
            });
        (full_data, accept)
    }

    // average all MC data and compute error bars
    fn process_monte_carlo_results(
        mc_results: &HashMap<String, Vec<OperatorValue>>,
        block_size: usize,
    ) -> Result<
        (
            HashMap<String, OperatorValue>,
            HashMap<String, OperatorValue>,
        ),
        Error,
    > {
        use OperatorValue::Scalar;
        // computes averages of all components of concatenated data
        let averages = mc_results
            .iter()
            .map(|(name, samples)| (name.to_string(), Self::mean(&samples)))
            .collect::<HashMap<_, _>>();
        // blocking error computation
        let errors = mc_results
            .iter()
            .map(|(name, samples)| {
                // mean of this quantity
                let mean = averages
                    .get(name)
                    .expect("Given observable not present in results");
                // split the data into blocks
                let blocks = samples.chunks(block_size);
                let nblocks = blocks.len();
                // compute averages of each block
                let block_means = blocks.map(Self::mean);
                // compute square of block averages
                let block_mean_square =
                    Self::mean(&block_means.clone().map(|x| &x * &x).collect::<Vec<_>>());
                // compute error
                let error = ((block_mean_square - mean * mean) / Scalar((nblocks - 1) as f64))
                    .map(f64::sqrt);
                (name.to_string(), error)
            })
            .collect::<HashMap<_, _>>();
        Ok((averages, errors))
    }

    fn mean(vec: &[OperatorValue]) -> OperatorValue {
        use OperatorValue::*;
        vec.iter().enumerate().fold(Scalar(0.0), |a, (n, b)| {
            &a + &((b - &a) / Scalar((n + 1) as f64))
        })
    }
}
