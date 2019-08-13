use std::collections::HashMap;

#[macro_use]
extern crate itertools;
use rand::{StdRng, SeedableRng};
use gnuplot::{AxesCommon, Caption, Color, Figure, FillAlpha};
use ndarray::{array, Array1, Array2};

#[allow(unused_imports)]
use mole::prelude::*;

#[derive(Clone)]
struct Logger {
    block_size: usize,
}
impl Log for Logger {
    fn log(&mut self, data: &HashMap<String, Vec<OperatorValue>>) -> String {
        let energy = data
            .get("Energy")
            .unwrap()
            .chunks(self.block_size)
            .last()
            .unwrap()
            .iter()
            .fold(0.0, |a, b| a + b.get_scalar().unwrap())
            / self.block_size as f64;
        format!("\tBlock energy:    {:.8}", energy)
    }
}

// Number of VMC iterations
static NITERS: usize = 20;
// Number of threads
static NWORKERS: usize = 8;
// Total number of MC samples, distributed over workers
// samples per worker is TOTAL_SAMPLES / NWORKERS
static TOTAL_SAMPLES: usize = 5000;
// Block size for blocking analysis. Effective number of
// samples, assuming unit correlation time:
// TOTAL_SAMPLES - BLOCK_SIZE * NWORKERS
static BLOCK_SIZE: usize = 50;
// Number of Jastrow factor parameters
static NPARM_JAS: usize = 2;

fn main() {
    let width = 1.0;

    // H2 equilibrium geometry
    let ion_pos = array![[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];

    // setup STO basis set
    let basis_set = Hydrogen1sBasis::new(ion_pos.clone(), vec![width, width]);

    // construct orbitals
    let orbitals = vec![
        Orbital::new(array![[1.0], [1.0]], basis_set.clone()),
        Orbital::new(array![[1.0], [1.0]], basis_set.clone()),
    ];

    // construct Jastrow-Slater wave function
    let wave_function = JastrowSlater::new(
        Array1::zeros(NPARM_JAS), // Jastrow factor parameters
        orbitals.clone(),
        0.001, // scale distance
        1,     // number of electrons with spin up
    )
    .expect("Bad wave function");

    // run optimization for two different optimizers
    let (energies_sr, errors_sr) = optimize_wave_function(
        &ion_pos,
        wave_function.clone(),
        StochasticReconfiguration::new(0.25),
    );
    let (energies_sd, errors_sd) =
        optimize_wave_function(&ion_pos, wave_function.clone(), SteepestDescent::new(0.05));

    // Plot the results
    plot_results(
        &[energies_sr, energies_sd],
        &[errors_sr, errors_sd],
        &["blue", "red"],
        &["Stochastic Refonfiguration", "Steepest Descent"],
    );
}

fn optimize_wave_function<O: Optimizer + Send + Sync + Clone>(
    ion_pos: &Array2<f64>,
    wave_function: JastrowSlater<Hydrogen1sBasis>,
    opt: O,
) -> (Array1<f64>, Array1<f64>) {
    //  hamiltonian operator
    let hamiltonian = ElectronicHamiltonian::from_ions(ion_pos.clone(), array![1, 1]);

    let obs = operators! {
        "Energy" => hamiltonian,
        "Parameter gradient" => ParameterGradient,
        "Wavefunction value" => WavefunctionValue
    };

    let (_wave_function, energies, errors) = {
        let sampler = Sampler::new(
            wave_function,
            metropolis::MetropolisDiffuse::from_rng(0.25, StdRng::from_seed([0_u8; 32])),
            &obs,
        )
        .expect("Bad initial configuration");

        // Construct the VMC runner, with Stochastic reconfiguration as optimizer
        // and an empty Logger so no output is given during each VMC iteration
        let vmc_runner = VmcRunner::new(
            sampler,
            opt,
            Logger {
                block_size: BLOCK_SIZE,
            },
        );

        // Actually run the VMC optimization
        vmc_runner.run_optimization(NITERS, TOTAL_SAMPLES, BLOCK_SIZE, NWORKERS)
    }
    .expect("VMC optimization failed");

    (energies, errors)
}

fn plot_results(
    energies: &[Array1<f64>],
    errors: &[Array1<f64>],
    colors: &[&str],
    labels: &[&str],
) {
    let niters = energies[0].len();
    let iters: Vec<_> = (0..niters).collect();
    let exact = vec![-1.175; niters];

    let mut fig = Figure::new();
    let axes = fig.axes2d();
    for (energy, error, color, label) in izip!(energies, errors, colors, labels) {
        axes.fill_between(
            &iters,
            &(energy - error),
            &(energy + error),
            &[Color(color), FillAlpha(0.1)],
        )
        .lines(&iters, energy, &[Caption(label), Color(color)]);
    }
    axes.lines(
        &iters,
        &exact,
        &[Caption("Best ground state energy, H2"), Color("black")],
    )
    .set_x_label("Iteration", &[])
    .set_y_label("VMC Energy (Hartree)", &[])
    .set_x_grid(true)
    .set_y_grid(true);

    fig.show();
}
