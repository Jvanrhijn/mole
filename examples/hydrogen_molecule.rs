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
    traits::{Log, MonteCarloResult},
    Runner, Sampler,
};
use operator::{
    ElectronicHamiltonian, Operator, OperatorValue, ParameterGradient, WavefunctionValue,
};
use optimize::{Optimize, Optimizer, StochasticReconfiguration, SteepestDescent, MomentumDescent};
use wavefunction::{Cache, Differentiate, Function, JastrowSlater, Orbital, WaveFunction};

// testing ground for VMC
#[derive(Clone)]
struct EmptyLogger;
impl Log for EmptyLogger {
    fn log(&mut self, _data: &HashMap<String, Vec<OperatorValue>>) -> String {
        String::new()
    }
}

struct VmcResult {
    pub energy: f64,
    pub energy_error: f64,
    pub energy_grad: Array1<f64>,
    pub parameter_grads: Array2<f64>,
    pub wf_values: Array1<f64>,
}

struct VmcRunner<H, T, L, O> {
    wave_function: T,
    logger: L,
    optimizer: O,
    hamiltonian: H,
}

impl<H, T, L, O> VmcRunner<H, T, L, O>
where
    O: Optimizer + Send + Clone + 'static,
    T: WaveFunction
        + Differentiate
        + Function<f64, D = Ix2>
        + Cache
        + Optimize
        + Clone
        + Send
        + 'static,
    L: Log + Clone + Send + 'static,
    H: Operator<T> + Clone + Send + 'static,
{
    pub fn new(wave_function: T, hamiltonian: H, optimizer: O, logger: L) -> Self {
        Self {
            wave_function,
            logger,
            optimizer,
            hamiltonian,
        }
    }

    #[allow(dead_code)]
    pub fn run_optimization<M>(
        mut self,
        iters: usize,
        total_samples: usize,
        block_size: usize,
        metropoles: Vec<M>,
    ) -> (T, Array1<f64>, Array1<f64>)
    where
        M: Metropolis<T> + Clone + Send + 'static,
        <M as Metropolis<T>>::R: RngCore,
    {
        let workers = metropoles.len();

        let steps = total_samples / workers;

        let nparm = self.wave_function.num_parameters();

        let mut energies = Vec::new();
        let mut energy_errs = Vec::new();

        for t in 0..iters {
            let (tx, rx) = mpsc::channel();

            let metrops = metropoles.clone();

            for (worker, metropolis) in metrops.into_iter().enumerate() {
                let sender = tx.clone();

                let wf = self.wave_function.clone();

                let metropolis = metropolis.clone();

                let hamiltonian = self.hamiltonian.clone();

                // TODO: only allow one thread to log output
                let logger = self.logger.clone();

                thread::spawn(move || {
                    let mut sampler = Sampler::new(wf, metropolis);
                    sampler.add_observable("Energy", hamiltonian);
                    sampler.add_observable("Parameter gradient", ParameterGradient);
                    sampler.add_observable("Wavefunction value", WavefunctionValue);

                    let runner = Runner::new(sampler, logger);

                    let result = runner.run(steps, block_size);

                    sender
                        .send(result)
                        .expect("Failed to send Monte carlo result over channel");
                });
            }

            let mut results = Vec::new();
            for _ in 0..workers {
                results.push(rx.recv().unwrap());
            }

            let VmcResult {
                energy,
                energy_error,
                energy_grad,
                parameter_grads,
                wf_values,
            } = Self::process_monte_carlo_results(&results);

            energies.push(energy);
            energy_errs.push(energy_error);

            let deltap =
                self.optimizer
                    .compute_parameter_update(&(energy_grad, wf_values, parameter_grads));

            self.wave_function.update_parameters(&deltap);

            println!("Energy:      {:.8} +/- {:.8}", energy, energy_error);
        }

        (
            self.wave_function,
            Array1::from_vec(energies),
            Array1::from_vec(energy_errs),
        )
    }

    fn process_monte_carlo_results(mc_results: &[MonteCarloResult<T>]) -> VmcResult {
        // computes energy, energy error, energy gradient over several parallel MC runs
        let nparm = mc_results[0].wave_function.num_parameters();
        let mut full_data: HashMap<String, Vec<OperatorValue>> = HashMap::new();
        // concatenate all MC worker results
        for MonteCarloResult { data, .. } in mc_results {
            for (key, data) in data.iter() {
                full_data
                    .entry(key.to_string())
                    .or_insert_with(|| Vec::new())
                    .append(&mut (data.clone()));
            }
        }

        let energies = Array1::from_vec(
            full_data
                .get("Energy")
                .unwrap()
                .iter()
                .map(|x| *x.get_scalar().unwrap())
                .collect(),
        );

        let wf_values = Array1::from_vec(
            full_data
                .get("Wavefunction value")
                .unwrap()
                .iter()
                .map(|x| *x.get_scalar().unwrap())
                .collect(),
        );

        let energy = *energies.mean_axis(Axis(0)).first().unwrap();

        let nsamples = energies.len();
        let mut parameter_grads = Array2::<f64>::zeros((nsamples, nparm));

        for (i, pgrad) in full_data
            .get("Parameter gradient")
            .unwrap()
            .iter()
            .enumerate()
        {
            let mut pgrad_slice = parameter_grads.slice_mut(s![i, ..]);
            pgrad_slice += pgrad.get_vector().unwrap();
        }

        // compute energy gradient as <E>_i = 2<psi_i / psi (E_L - <E>)>
        let mut local_energy_grad = Array2::<f64>::zeros((nsamples, nparm));
        Zip::from(local_energy_grad.genrows_mut())
            .and(parameter_grads.genrows())
            .and(&energies)
            .and(&wf_values)
            .apply(|mut ge, psi_i, &e, &psi| ge += &(2.0 * &(&psi_i / psi) * (e - energy)));

        VmcResult {
            energy,
            energy_error: *energies.std_axis(Axis(0), 0.0).first().unwrap()
                / (nsamples as f64).sqrt(),
            energy_grad: local_energy_grad.mean_axis(Axis(0)),
            parameter_grads,
            wf_values,
        }
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
    const NITERS: usize = 33;

    // use 8 threads
    const NWORKERS: usize = 8;

    // Sample 10_000 data points across all workers
    const TOTAL_SAMPLES: usize = 10_000;

    // use a block size of 10
    const BLOCK_SIZE: usize = 10;

    // Use 2 Jastrow factor parameters (b2 and b3)
    const NPARM_JAS: usize = 2;

    // SR step size
    const STEP_SIZE: f64 = 0.1;
    const MOMENTUM_PARAMETER: f64 = 0.01;

    // construct Jastrow-Slater wave function
    let wave_function = JastrowSlater::new(
        Array1::zeros(NPARM_JAS),
        orbitals.clone(),
        0.001, // scale distance
        1,     // number of electrons with spin up
    );

    // Construct the VMC runner, with Stochastic reconfiguration as optimizer
    // and an empty Logger so no output is given during each VMC iteration
    let vmc_runner = VmcRunner::new(
        wave_function,
        hamiltonian,
        MomentumDescent::new(STEP_SIZE, MOMENTUM_PARAMETER, NPARM_JAS),
        //SteepestDescent::new(STEP_SIZE),
        //StochasticReconfiguration::new(STEP_SIZE),
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
        v
    };

    // Actually run the VMC optimization
    let (wave_function, energies, errors) =
        vmc_runner.run_optimization(NITERS, TOTAL_SAMPLES, BLOCK_SIZE, metrops);

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