use std::collections::HashMap;

#[macro_use]
extern crate itertools;
use gnuplot::{AxesCommon, Caption, Color, Figure, FillAlpha};
use ndarray::{Array1, Array2, array};
use rand::{StdRng, SeedableRng};

use mole::prelude::*;

#[derive(Clone)]
struct EmptyLogger {
    block_size: usize,
}
impl Log for EmptyLogger {
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

static NITERS: usize = 20;
static NWORKERS: usize = 8;
static TOTAL_SAMPLES: usize = 10_000;
static BLOCK_SIZE: usize = 50;
static NPARM_JAS: usize = 2;

fn main() {
    let width = 0.6;
    // setup basis set
    let ion_pos = array![[0.0, 0.0, 0.0]];

    let basis_set = Hydrogen1sBasis::new(ion_pos.clone(), vec![width]);

    // construct orbitals
    let orbitals = vec![
        Orbital::new(array![[1.0]], basis_set.clone()),
        Orbital::new(array![[1.0]], basis_set.clone()),
    ];

    // construct Jastrow-Slater wave function
    let wave_function = JastrowSlater::new(
        Array1::zeros(NPARM_JAS), // Jastrow factor parameters
        orbitals.clone(),
        0.001, // scale distance
        1,     // number of electrons with spin up
    )
    .expect("Bad wave function");

    // run optimization
    let (energies_sr, errors_sr) = optimize_wave_function(
        &ion_pos,
        wave_function.clone(),
        StochasticReconfiguration::new(10.0),
    );
    let (energies_sd, errors_sd) =
        optimize_wave_function(&ion_pos, wave_function.clone(), SteepestDescent::new(0.001));

    // Plot the results
    plot_results(
        &[energies_sr, energies_sd],
        &[errors_sr, errors_sd],
        &["blue", "red"],
        &["SR", "SD"],
    );
}

fn optimize_wave_function<O: Optimizer + Send + Sync + Clone>(
    ion_pos: &Array2<f64>,
    wave_function: JastrowSlater<Hydrogen1sBasis>,
    opt: O,
) -> (Array1<f64>, Array1<f64>) {
    //  hamiltonian operator
    let hamiltonian = ElectronicHamiltonian::from_ions(ion_pos.clone(), array![2]);

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
            //OnlineLbfgs::new(0.1, 10, NPARM_JAS),
            //NesterovMomentum::new(0.01, 0.00001, NPARM_JAS),
            opt,
            EmptyLogger {
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
    let exact = vec![-2.903; niters];

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
        &[Caption("Best ground state energy, Helium"), Color("black")],
    )
    .set_x_label("Iteration", &[])
    .set_y_label("VMC Energy (Hartree)", &[])
    .set_x_grid(true)
    .set_y_grid(true);

    fig.show();
}
