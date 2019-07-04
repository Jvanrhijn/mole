use rayon::iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};
use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;
#[macro_use]
extern crate ndarray;
use rand::{RngCore, SeedableRng, StdRng};

use gnuplot::{AxesCommon, Caption, Color, Figure, FillAlpha};

#[macro_use]
extern crate itertools;

use ndarray::{Array1, Array2, Axis, Ix2, Zip};

use basis::Hydrogen1sBasis;
use metropolis::{Metropolis, MetropolisDiffuse};
use montecarlo::{
    traits::MonteCarloSampler,
    traits::{Log, MonteCarloResult},
    Runner, Sampler,
};
use operator::{ElectronicHamiltonian, Operator, OperatorValue};
use optimize::{
    MomentumDescent, NesterovMomentum, OnlineLbfgs, Optimize, Optimizer, SteepestDescent,
    StochasticReconfiguration,
};
//use optimize::{
//    MomentumDescent, NesterovMomentum, OnlineLbfgs, Optimize, Optimizer, SteepestDescent,
//    StochasticReconfiguration,
//};
use errors::Error;

use wavefunction::{JastrowSlater, Orbital};
use wavefunction_traits::{Cache, Differentiate, Function, WaveFunction};

// testing ground for VMC
struct ParameterGradient;

impl<T: Optimize + Cache> Operator<T> for ParameterGradient {
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue, Error> {
        Ok(OperatorValue::Vector(wf.parameter_gradient(cfg))
            * OperatorValue::Scalar(wf.current_value().0))
    }
}

#[derive(Copy, Clone)]
struct WavefunctionValue;

impl<T: Cache> Operator<T> for WavefunctionValue {
    fn act_on(&self, wf: &T, _cfg: &Array2<f64>) -> Result<OperatorValue, Error> {
        // need to square this, since "local value" is operator product / wave function value
        Ok(OperatorValue::Scalar(wf.current_value().0.powi(2)))
    }
}

#[derive(Clone)]
struct EmptyLogger;
impl Log for EmptyLogger {
    fn log(&mut self, _data: &HashMap<String, Vec<OperatorValue>>) -> String {
        String::new()
    }
}

struct VmcRunner<S, L, O> {
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
    <S as MonteCarloSampler>::Seed: From<[u8; 32]>,
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
    ) -> (T, Array1<f64>, Array1<f64>)
    {
        let steps = total_samples / nworkers;

        let mut energies = Vec::new();
        let mut energy_errs = Vec::new();

        for _ in 0..iters {

            let samplers = vec![self.sampler.clone(); nworkers];
 
            let results: Vec<_> = samplers.into_par_iter().enumerate().map(|(worker, mut sampler)| {

                sampler.reseed_rng([worker as u8; 32].into());

                let logger = self.logger.clone();

                let runner = Runner::new(sampler, logger);

                runner.run(steps, block_size)

            }).collect();

            let mc_data = Self::concatenate_worker_data(&results);

            let (averages, variance) = Self::process_monte_carlo_results(&mc_data);

            energies.push(*averages["Energy"].get_scalar().unwrap());
            energy_errs
                .push((variance["Energy"].get_scalar().unwrap() / total_samples as f64).sqrt());

            let deltap = self.optimizer.compute_parameter_update(self.sampler.wave_function().parameters(), &averages, &mc_data);

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
// end testing ground for vmc

fn main() {
    // First, set up the problem

    // Equilibrium H2 geometry
    let ion_positions = array![[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];

    // Use STO basis set, with one basis function centered on
    // each proton
    let basis_set = Hydrogen1sBasis::new(ion_positions.clone(), vec![1.0]);

    // Construct two orbitals from basis, both equally centered
    // on each atom -- allowed since the electrons have opposite spin
    // according to Hund's rule
    let orbitals = vec![
        Orbital::new(array![[1.0], [1.0]], basis_set.clone()),
        Orbital::new(array![[1.0], [1.0]], basis_set.clone()),
    ];

    // Build the Electronic hamiltonian from the ion positions
    let hamiltonian = ElectronicHamiltonian::from_ions(ion_positions, array![1, 1]);

    // Set VMC parameters
    // use 100 iterations
    const NITERS: usize = 20;

    // use 8 threads
    const NWORKERS: usize = 4;

    // Sample 10_000 data points across all workers
    const TOTAL_SAMPLES: usize = 10_000;

    // use a block size of 10
    const BLOCK_SIZE: usize = 10;

    // Use 2 Jastrow factor parameters (b2 and b3)
    const NPARM_JAS: usize = 2;

    // SR step size
    const STEP_SIZE: f64 = 0.1;
    const MOMENTUM_PARAMETER: f64 = 0.1;

    // construct Jastrow-Slater wave function
    let wave_function = JastrowSlater::new(
        Array1::zeros(NPARM_JAS),
        orbitals.clone(),
        0.001, // scale distance
        1,     // number of electrons with spin up
    );

    let mut obs = HashMap::new();
    obs.insert(
        "Energy".to_string(),
        Box::new(hamiltonian) as Box<dyn Operator<JastrowSlater<Hydrogen1sBasis>> + Send + Sync>,
    );    obs.insert(
        "Parameter gradient".to_string(),
        Box::new(ParameterGradient) as Box<dyn Operator<JastrowSlater<Hydrogen1sBasis>> + Send + Sync>,
    );
    obs.insert(
        "Wavefunction value".to_string(),
        Box::new(WavefunctionValue) as Box<dyn Operator<JastrowSlater<Hydrogen1sBasis>> + Send + Sync>,
    );

    let (wave_function, energies, errors) = {
        let sampler = Sampler::new(wave_function, metropolis::MetropolisDiffuse::from_rng(0.1, StdRng::from_seed([0_u8; 32])), &obs);

        // Construct the VMC runner, with Stochastic reconfiguration as optimizer
        // and an empty Logger so no output is given during each VMC iteration
        let vmc_runner = VmcRunner::new(
            sampler,
            //OnlineLbfgs::new(0.05, 5, NPARM_JAS),
            //NesterovMomentum::new(STEP_SIZE, MOMENTUM_PARAMETER, NPARM_JAS),
            SteepestDescent::new(0.05),
            //StochasticReconfiguration::new(0.05),
            EmptyLogger,
        );

        // Construct a metropolis object for each worker, so we know
        // the seed given to each worker
        let metrops = {
            let mut v = vec![];
            for m in 0..NWORKERS {
                v.push(MetropolisDiffuse::from_rng(
                    0.2,
                    StdRng::from_seed([m as u8; 32]),
                ))
            }
        };

        // Actually run the VMC optimization
        vmc_runner.run_optimization(NITERS, TOTAL_SAMPLES, BLOCK_SIZE, NWORKERS)
    };

    // Plot the results
    plot_results(energies.as_slice().unwrap(), errors.as_slice().unwrap());
}

fn plot_results(energy: &[f64], error: &[f64]) {
    let niters = energy.len();
    let iters: Vec<_> = (0..niters).collect();
    let exact = vec![-1.175; niters];

    let mut fig = Figure::new();
    fig.axes2d()
        .y_error_bars(&iters, energy, error, &[Color("blue")])
        .lines(
            &iters,
            energy,
            &[Caption("VMC Energy of H2"), Color("blue")],
        )
        .lines(
            &iters,
            &exact,
            &[Caption("Best ground state energy"), Color("red")],
        )
        .set_x_label("Iteration", &[])
        .set_y_label("VMC Energy (Hartree)", &[])
        .set_x_grid(true)
        .set_y_grid(true);

    fig.show();
}
