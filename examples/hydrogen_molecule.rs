use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;
#[macro_use]
extern crate ndarray;
use rand::{RngCore, SeedableRng, StdRng};

use gnuplot::{AxesCommon, Caption, Color, Figure, FillAlpha};

#[macro_use]
extern crate itertools;

use ndarray::{Array1, Array2, Axis, Ix2};
use ndarray_linalg::SolveH;

use basis::Hydrogen1sBasis;
use metropolis::{Metropolis, MetropolisDiffuse};
use montecarlo::{
    traits::{Log, MonteCarloResult, MonteCarloSampler},
    Runner, Sampler,
};
use operator::{
    ElectronicHamiltonian, Operator, OperatorValue, ParameterGradient, WavefunctionValue,
};
use optimize::{Optimize, Optimizer};
use wavefunction::{Cache, Differentiate, Function, JastrowSlater, Orbital, WaveFunction};

// testing ground for VMC
#[derive(Clone)]
struct EmptyLogger;
impl Log for EmptyLogger {
    fn log(&mut self, _data: &HashMap<String, Vec<OperatorValue>>) -> String {
        String::new()
    }
}

#[derive(Clone)]
struct StochasticReconfiguration {
    step_size: f64,
}

impl StochasticReconfiguration {
    fn construct_sr_matrix(parm_grad: &[Array1<f64>], wf_values: &[f64]) -> Array2<f64> {
        let nparm = parm_grad[0].len();
        let nsamples = parm_grad.len();

        // construct the stochastic reconfiguration matrix
        let mut sr_mat = Array2::<f64>::zeros((nparm, nparm));

        // build array2 of o_i values
        let mut sr_o = Array2::<f64>::zeros((nsamples, nparm));
        for n in 0..nsamples {
            for i in 0..nparm {
                sr_o[[n, i]] = parm_grad[n][i] / wf_values[n];
            }
        }

        // add the <Ok Ol> term to sr_mat
        for n in 0..nsamples {
            sr_mat += &(outer_product(
                &sr_o.slice(s![n, ..]).to_owned(),
                &sr_o.slice(s![n, ..]).to_owned(),
            ) / nsamples as f64);
        }

        let sr_o_avg = sr_o.mean_axis(Axis(0));

        // subtract <Ok><Ol>
        for i in 0..nparm {
            for j in 0..nparm {
                sr_mat -= sr_o_avg[i] * sr_o_avg[j];
            }
        }

        sr_mat //- &sr_o_avg_mat2
    }
}

impl Optimizer for StochasticReconfiguration {
    fn compute_parameter_update(
        &mut self,
        (energy_grad, wf_values, grad_parm): &(Array1<f64>, Vec<f64>, Vec<Array1<f64>>),
    ) -> Array1<f64> {
        let sr_matrix = StochasticReconfiguration::construct_sr_matrix(&grad_parm, &wf_values);
        self.step_size * sr_matrix.solveh_into(-0.5 * energy_grad).unwrap()
    }
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

                    let MonteCarloResult { data, .. } = runner.run(steps, block_size);

                    let energy_data = Array1::<f64>::from_vec(
                        data.get("Energy")
                            .unwrap()
                            .iter()
                            .map(|x| *x.get_scalar().unwrap())
                            .collect::<Vec<_>>(),
                    );

                    // Retrieve mean values of energy over run
                    let energy = *energy_data.mean_axis(Axis(0)).first().unwrap();
                    let energy_err = *energy_data.std_axis(Axis(0), 0.0).first().unwrap()
                        / ((steps - block_size) as f64).sqrt();

                    let par_grads: Vec<_> = data
                        .get("Parameter gradient")
                        .unwrap()
                        .iter()
                        .map(|x| x.get_vector().unwrap().clone())
                        .collect();

                    let local_energy: Vec<_> = data
                        .get("Energy")
                        .unwrap()
                        .iter()
                        .map(|x| *x.get_scalar().unwrap())
                        .collect();
                    let wf_values: Vec<_> = data
                        .get("Wavefunction value")
                        .unwrap()
                        .iter()
                        .map(|x| *x.get_scalar().unwrap())
                        .collect();

                    // obtain the energy gradient
                    let local_energy_grad = izip!(par_grads.iter(), local_energy, wf_values.iter())
                        .map(|(psi_i, el, psi)| 2.0 * psi_i / *psi * (el - energy))
                        .collect::<Vec<Array1<f64>>>();

                    let energy_grad = local_energy_grad
                        .iter()
                        .fold(Array1::zeros(nparm), |a, b| a + b)
                        / (steps - block_size) as f64;

                    // TODO: refactor into just sending MonteCarloResult, do data processing
                    // on main thread
                    sender
                        .send((energy, energy_grad, energy_err, par_grads, wf_values))
                        .expect("Failed to send Monte carlo result over channel");
                });
            }

            let mut energy = 0.0;
            let mut energy_err_sq = 0.0;
            let mut energy_grad = Array1::<f64>::zeros(nparm);
            let mut wf_values = Vec::new();
            let mut grad_parm = Vec::new();

            for _ in 0..workers {
                let (e, eg, e_err, mut gparm, mut wfv) = rx.recv().unwrap();
                energy += e / workers as f64;
                energy_err_sq += e_err.powi(2);
                energy_grad += &(eg / workers as f64);
                wf_values.append(&mut wfv);
                grad_parm.append(&mut gparm)
            }

            let energy_err = energy_err_sq.sqrt() / workers as f64;

            energies.push(energy);
            energy_errs.push(energy_err);

            let deltap =
                self.optimizer
                    .compute_parameter_update(&(energy_grad, wf_values, grad_parm));

            self.wave_function.update_parameters(&deltap);

            println!("Energy:      {:.8} +/- {:.8}", energy, energy_err);
        }

        (
            self.wave_function,
            Array1::from_vec(energies),
            Array1::from_vec(energy_errs),
        )
    }
}
// end testing ground for vmc

fn main() {
    // First, set up the problem

    // Equilibrium H2 geometry
    let ion_positions = array![[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];

    // Use STO basis set, with one basis function
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
    const NITERS: usize = 100;
    // use 8 threads
    const NWORKERS: usize = 8;

    // Sample 10_000 data points across all workers
    const TOTAL_SAMPLES: usize = 10_000;

    // use a block size of 10
    let block_size = 10;

    // Use 2 Jastrow factor parameters (b2 and b3)
    const NPARM_JAS: usize = 2;

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
        StochasticReconfiguration { step_size: 0.01 },
        EmptyLogger,
    );

    // Construct a metropolis object for each worker, so we know
    // the seed given to each worker
    let metrops = {
        let mut v = vec![];
        for m in 0..NWORKERS {
            v.push(MetropolisDiffuse::from_rng(
                0.1,
                StdRng::from_seed([m as u8; 32]),
            ))
        }
        v
    };

    // Actually run the VMC optimization
    let (wave_function, energies, errors) =
        vmc_runner.run_optimization(NITERS, TOTAL_SAMPLES, block_size, metrops);

    // Plot the results
    let iters: Vec<_> = (0..NITERS).collect();
    let exact = vec![-1.175; NITERS];

    let mut fig = Figure::new();
    fig.axes2d()
        .fill_between(
            &iters,
            &(&energies - &errors),
            &(&energies + &errors),
            &[Color("blue"), FillAlpha(0.1)],
        )
        .lines(
            &iters,
            &energies,
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

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let mut result = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}
