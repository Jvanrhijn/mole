use std::collections::HashMap;
#[macro_use]
extern crate ndarray;
use rand::{SeedableRng, StdRng};

use gnuplot::{AxesCommon, Caption, Color, Figure, FillAlpha};

use ndarray::Array1;

use basis::Hydrogen1sBasis;
use montecarlo::{traits::Log, Sampler};
use operator::{ElectronicHamiltonian, OperatorValue};
use optimize::{
    MomentumDescent, NesterovMomentum, OnlineLbfgs, SteepestDescent, StochasticReconfiguration,
};
use vmc::{ParameterGradient, VmcRunner, WavefunctionValue};

use wavefunction::{JastrowSlater, Orbital};
#[macro_use]
extern crate util;

#[derive(Clone)]
struct EmptyLogger;
impl Log for EmptyLogger {
    fn log(&mut self, _data: &HashMap<String, Vec<OperatorValue>>) -> String {
        String::new()
    }
}

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
    const NITERS: usize = 25;

    // threads to use
    const NWORKERS: usize = 4;

    // Sample data points across all workers
    const TOTAL_SAMPLES: usize = 10_000;

    // use a block size of 10
    const BLOCK_SIZE: usize = 100;

    // Use 2 Jastrow factor parameters (b2 and b3)
    const NPARM_JAS: usize = 2;

    // construct Jastrow-Slater wave function
    let wave_function = JastrowSlater::new(
        Array1::zeros(NPARM_JAS),
        orbitals.clone(),
        0.001, // scale distance
        1,     // number of electrons with spin up
    );

    let obs = operators! {
        "Energy" => hamiltonian,
        "Parameter gradient" => ParameterGradient,
        "Wavefunction value" => WavefunctionValue
    };

    let (wave_function, energies, errors) = {
        let sampler = Sampler::new(
            wave_function,
            metropolis::MetropolisDiffuse::from_rng(0.1, StdRng::from_seed([0_u8; 32])),
            &obs,
        );

        // Construct the VMC runner, with Stochastic reconfiguration as optimizer
        // and an empty Logger so no output is given during each VMC iteration
        let vmc_runner = VmcRunner::new(
            sampler,
            //OnlineLbfgs::new(0.5, 5, NPARM_JAS),
            //NesterovMomentum::new(STEP_SIZE, MOMENTUM_PARAMETER, NPARM_JAS),
            //SteepestDescent::new(0.01),
            StochasticReconfiguration::new(0.05),
            EmptyLogger,
        );

        // Actually run the VMC optimization
        vmc_runner.run_optimization(NITERS, TOTAL_SAMPLES, BLOCK_SIZE, NWORKERS)
    }
    .unwrap();

    // Plot the results
    plot_results(&energies, &errors);
}

fn plot_results(energy: &Array1<f64>, error: &Array1<f64>) {
    let niters = energy.len();
    let iters: Vec<_> = (0..niters).collect();
    let exact = vec![-1.175; niters];

    let mut fig = Figure::new();
    fig.axes2d()
        .lines(
            &iters,
            energy,
            &[Caption("VMC Energy of H2"), Color("blue")],
        )
        .fill_between(
            &iters,
            &(energy - error),
            &(energy + error),
            &[Color("blue"), FillAlpha(0.1)]
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
