use std::collections::HashMap;

use montecarlo::traits::{Log, MonteCarloResult, MonteCarloSampler};
use montecarlo::Runner;
use operator::OperatorValue;
use optimize::{Optimize, Optimizer};
use wavefunction_traits::{Cache, Differentiate, Function, WaveFunction};

use ndarray::{Array1, Ix2};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

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
    O: Optimizer + Send + Sync + Clone + 'static,
    T: WaveFunction
        + Differentiate
        + Function<f64, D = Ix2>
        + Cache
        + Optimize
        + Clone
        + Send
        + Sync
        + 'static,
    L: Log + Clone + Send + Sync + 'static,
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
    ) -> (T, Array1<f64>, Array1<f64>) {
        let steps = total_samples / nworkers;

        let mut energies = Vec::new();
        let mut energy_errs = Vec::new();

        for _ in 0..iters {
            let samplers = vec![self.sampler.clone(); nworkers];

            // produce rng seeds for threads
            let seeds: Vec<_> = (0..nworkers).map(|_| self.sampler.generate_seed()).collect();

            let results: Vec<_> = samplers
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

            let mc_data = Self::concatenate_worker_data(&results);

            let (averages, variance) = Self::process_monte_carlo_results(&mc_data);

            energies.push(*averages["Energy"].get_scalar().unwrap());
            energy_errs
                .push((variance["Energy"].get_scalar().unwrap() / total_samples as f64).sqrt());

            let deltap = self.optimizer.compute_parameter_update(
                self.sampler.wave_function().parameters(),
                &averages,
                &mc_data,
            );

            self.sampler.wave_function_mut().update_parameters(&deltap);

            println!(
                "Energy:      {:.8} +/- {:.9}",
                energies.last().unwrap(),
                energy_errs.last().unwrap()
            );
        }

        (
            self.sampler.wave_function().clone(),
            Array1::from_vec(energies),
            Array1::from_vec(energy_errs),
        )
    }

    fn concatenate_worker_data(
        worker_data: &Vec<MonteCarloResult<T>>,
    ) -> HashMap<String, Vec<OperatorValue>> {
        // computes energy, energy error, energy gradient over several parallel MC runs
        let mut full_data: HashMap<String, Vec<OperatorValue>> = HashMap::new();
        // concatenate all MC worker results
        worker_data
            .iter()
            .for_each(|MonteCarloResult { data, .. }| {
                data.iter().for_each(|(key, data)| {
                    full_data
                        .entry(key.to_string())
                        .or_insert_with(|| Vec::new())
                        .append(&mut (data.clone()));
                })
            });
        full_data
    }

    // average all MC data and compute error bars
    fn process_monte_carlo_results(
        mc_results: &HashMap<String, Vec<OperatorValue>>,
    ) -> (
        HashMap<String, OperatorValue>,
        HashMap<String, OperatorValue>,
    ) {
        use OperatorValue::Scalar;
        // computes averages of all components of concatenated data
        let averages = mc_results
            .iter()
            .map(|(name, samples)| {
                (
                    name.to_string(),
                    samples.iter().enumerate().fold(Scalar(0.0), |a, (n, b)| {
                        &a + &((b - &a) / Scalar((n + 1) as f64))
                    }),
                )
            })
            .collect::<HashMap<_, _>>();
        // naive error computation, TODO: replace with better algorithm
        let variance = mc_results
            .iter()
            .map(|(name, samples)| {
                let mean_of_squares = samples
                    .iter()
                    .map(|x| x * x)
                    .enumerate()
                    .fold(Scalar(0.0), |a, (n, b)| {
                        &a + &((&b - &a) / Scalar((n + 1) as f64))
                    });
                let mean = averages.get(name).unwrap();
                let square_of_mean = mean * mean;
                (name.to_string(), mean_of_squares - square_of_mean)
            })
            .collect::<HashMap<_, _>>();
        (averages, variance)
    }
}
