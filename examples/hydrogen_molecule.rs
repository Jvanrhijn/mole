use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;
#[macro_use]
extern crate ndarray;
use rand::{SeedableRng, StdRng};

use gnuplot::{AxesCommon, Caption, Color, Figure, FillAlpha};

#[macro_use]
extern crate itertools;

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::SolveH;

use basis::Hydrogen1sBasis;
use metropolis::MetropolisDiffuse;
use montecarlo::{traits::Log, Runner, Sampler};
use operator::{ElectronicHamiltonian, OperatorValue, ParameterGradient, WavefunctionValue};
use wavefunction::{JastrowSlater, Orbital};

struct Logger {
    block_size: usize,
}

impl Logger {
    pub fn new(block_size: usize) -> Self {
        Self { block_size }
    }

    #[allow(dead_code)]
    fn compute_mean_and_block_avg(
        &self,
        name: &str,
        data: &HashMap<String, Vec<OperatorValue>>,
    ) -> (f64, f64) {
        let blocks = &data[name].chunks(self.block_size);

        let block_means = blocks.clone().map(|block| {
            block
                .iter()
                .fold(OperatorValue::Scalar(0.0), |a, b| a + b.clone())
                / OperatorValue::Scalar(block.len() as f64)
        });

        let quantity = *(block_means.clone().sum::<OperatorValue>()
            / OperatorValue::Scalar(block_means.len() as f64))
        .get_scalar()
        .unwrap();

        (quantity, *block_means.last().unwrap().get_scalar().unwrap())
    }
}

impl Log for Logger {
    fn log(&mut self, data: &HashMap<String, Vec<OperatorValue>>) -> String {
        //let (energy, energy_ba) = self.compute_mean_and_block_avg("Energy", data);
        //let (ke, ke_ba) = self.compute_mean_and_block_avg("Kinetic", data);
        //let (pe, pe_ba) = self.compute_mean_and_block_avg("Electron potential", data);

        //format!(
        //    "Energy: {:.5}  {:.5}    Kinetic: {:.5}    Electron Potential: {:.5}",
        //    energy, energy_ba, ke, pe
        //)
        String::new()
    }
}

fn main() {
    let ion_positions = array![[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];

    let basis_set = Hydrogen1sBasis::new(ion_positions.clone(), vec![1.0]);

    let orbitals = vec![
        Orbital::new(array![[1.0], [1.0]], basis_set.clone()),
        Orbital::new(array![[1.0], [1.0]], basis_set.clone()),
    ];

    let hamiltonian = ElectronicHamiltonian::from_ions(ion_positions, array![1, 1]);

    const NITERS: usize = 100;
    const NWORKERS: usize = 8;

    const TOTAL_SAMPLES: usize = 10_000;

    let block_size = 10;
    let steps = TOTAL_SAMPLES / NWORKERS;

    const NPARM_JAS: usize = 2;

    let mut energies = Array1::<f64>::zeros(NITERS);
    let mut errors = Array1::<f64>::zeros(NITERS);

    let mut jas_parm = Array1::zeros(NPARM_JAS);

    for t in 0..NITERS {
        // construct Jastrow-Slater wave function
        let wave_function = JastrowSlater::new(
            jas_parm.clone(), // Jastrow factor parameters
            orbitals.clone(),
            0.001, // scale distance
            1,     // number of electrons with spin up
        );

        let (tx, rx) = mpsc::channel();

        for worker in 0..NWORKERS {
            let sender = tx.clone();

            let wf = wave_function.clone();

            let jas_parm = jas_parm.clone();

            let hamiltonian = hamiltonian.clone();

            thread::spawn(move || {
                // setup metropolis algorithm/markov chain generator
                let metrop =
                    MetropolisDiffuse::from_rng(0.1, StdRng::from_seed([worker as u8; 32]));

                // construct sampler
                let mut sampler = Sampler::new(wf.clone(), metrop);
                sampler.add_observable("Hamiltonian", hamiltonian.clone());
                sampler.add_observable("Parameter gradient", ParameterGradient);
                sampler.add_observable("Wavefunction value", WavefunctionValue);

                // create MC runner
                let mut runner = Runner::new(sampler, Logger::new(block_size));

                // Run Monte Carlo integration
                runner.run(steps, block_size);

                let energy_data = Array1::<f64>::from_vec(
                    runner
                        .data()
                        .get("Hamiltonian")
                        .unwrap()
                        .iter()
                        .map(|x| *x.get_scalar().unwrap())
                        .collect::<Vec<_>>(),
                );

                // Retrieve mean values of energy over run
                let energy = *energy_data.mean_axis(Axis(0)).first().unwrap();
                let energy_err = *energy_data.std_axis(Axis(0), 0.0).first().unwrap()
                    / ((steps - block_size) as f64).sqrt();

                let par_grads: Vec<_> = runner
                    .data()
                    .get("Parameter gradient")
                    .unwrap()
                    .iter()
                    .map(|x| x.get_vector().unwrap().clone())
                    .collect();
                let local_energy: Vec<_> = runner
                    .data()
                    .get("Hamiltonian")
                    .unwrap()
                    .iter()
                    .map(|x| *x.get_scalar().unwrap())
                    .collect();
                let wf_values: Vec<_> = runner
                    .data()
                    .get("Wavefunction value")
                    .unwrap()
                    .iter()
                    .map(|x| *x.get_scalar().unwrap())
                    .collect();

                let sr_matrix = construct_sr_matrix(&par_grads, &wf_values);

                // obtain the energy gradient
                let local_energy_grad = izip!(par_grads, local_energy, wf_values)
                    .map(|(psi_i, el, psi)| 2.0 * psi_i / psi * (el - energy))
                    .collect::<Vec<Array1<f64>>>();

                let energy_grad = local_energy_grad
                    .iter()
                    .fold(Array1::zeros(jas_parm.len()), |a, b| a + b)
                    / (steps - block_size) as f64;

                sender
                    .send((sr_matrix, energy_grad, energy, energy_err))
                    .unwrap();
            });
        }

        let mut results: Vec<(Array2<f64>, Array1<f64>, f64, f64)> = Vec::new();

        //match rx.recv() {
        //    Ok(result) => results.push(result),
        //    Err(_) => panic!("Receive error")
        //};

        for _ in 0..NWORKERS {
            let result = rx.recv().unwrap();
            results.push(result);
        }

        // average over results from workers
        let (sr_matrix, energy_grad, energy, energy_err_sq) = results.into_iter().fold(
            (
                Array2::<f64>::zeros((NPARM_JAS, NPARM_JAS)),
                Array1::<f64>::zeros(NPARM_JAS),
                0.0,
                0.0,
            ),
            |(srmat, g, e, err), (srmat2, g2, e2, err2)| {
                (srmat + srmat2, g + g2, e + e2, err + err2.powi(2))
            },
        );

        let energy = energy / NWORKERS as f64;
        let sr_matrix = sr_matrix / NWORKERS as f64;
        let energy_grad = energy_grad / NWORKERS as f64;
        let energy_err = energy_err_sq.sqrt() / NWORKERS as f64;

        let sr_direction = sr_matrix.solveh_into(-0.5 * energy_grad).unwrap();

        energies[t] = energy;
        errors[t] = energy_err;
        println!("Energy:         {:.*} +/- {:.*}", 8, energy, 8, energy_err);

        ////println!("Exact ground state energy: -2.903");

        // do SR step
        let step_size = 0.1;
        jas_parm += &(step_size * sr_direction);

        //println!("\nSuggested new parameters: {}", jas_parm);
    }

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

fn covariance(a: &Array2<f64>) -> Array2<f64> {
    let shape = a.shape();
    let dim = shape[1];
    let nsamples = shape[0];
    let mut result = Array2::<f64>::zeros((dim, dim));
    let mut result2 = Array2::<f64>::zeros((dim, dim));
    let a_avg = a.mean_axis(Axis(0));
    let a_avg2 = a.mean_axis(Axis(0));
    for n in 0..nsamples {
        let a_slice = a.slice(s![n, ..]);
        result2 += &outer_product(&(&a_slice - &a_avg2), &(&a_slice - &a_avg2));
        for i in 0..dim {
            for j in 0..dim {
                result[[i, j]] += (a[[n, i]] - a_avg[i]) * (a[[n, j]] - a_avg[j]);
            }
        }
    }
    dbg!(&result - &result2);
    result / (nsamples as f64)
}

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
