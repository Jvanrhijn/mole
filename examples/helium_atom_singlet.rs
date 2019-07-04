use std::collections::HashMap;

use gnuplot::{AxesCommon, Caption, Color, Figure};
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate ndarray;
use basis::Hydrogen1sBasis;
use montecarlo::{
    Sampler,
    traits::Log,
};
use ndarray::Array1;
use operator::{ElectronicHamiltonian, OperatorValue};
use optimize::{SteepestDescent, NesterovMomentum, StochasticReconfiguration, OnlineLbfgs};
use rand::{SeedableRng, StdRng};
use wavefunction::{JastrowSlater, Orbital};
use vmc::{VmcRunner, ParameterGradient, WavefunctionValue};
#[macro_use]
extern crate util;

#[derive(Clone)]
struct EmptyLogger { block_size: usize }
impl Log for EmptyLogger {
    fn log(&mut self, data: &HashMap<String, Vec<OperatorValue>>) -> String {
        let energy = data.get("Energy").unwrap().chunks(self.block_size).last().unwrap().iter()
            .fold(0.0, |a, b| a + b.get_scalar().unwrap()) / self.block_size as f64;
        format!("\tLocal energy    {:.8}", energy)
    }
}

fn main() {
    let optimal_width = 0.5;
    // setup basis set
    let ion_pos = array![[0.0, 0.0, 0.0]];

    let basis_set = Hydrogen1sBasis::new(ion_pos.clone(), vec![optimal_width]);

    // construct orbitals
    let orbitals = vec![
        Orbital::new(array![[1.0]], basis_set.clone()),
        Orbital::new(array![[1.0]], basis_set.clone()),
    ];

    const NPARM_JAS: usize = 2;

    //  hamiltonian operator
    let hamiltonian = ElectronicHamiltonian::from_ions(ion_pos, array![2]);

    const NITERS: usize = 10;
    const NWORKERS: usize = 4;

    const TOTAL_SAMPLES: usize = 10_000;

    const BLOCK_SIZE: usize = 50;

    // construct Jastrow-Slater wave function
    let wave_function = JastrowSlater::new(
        Array1::zeros(NPARM_JAS), // Jastrow factor parameters
        orbitals.clone(),
        0.001, // scale distance
        1,     // number of electrons with spin up
    );

    let obs = operators!{
        "Energy" => hamiltonian,
        "Parameter gradient" => ParameterGradient,
        "Wavefunction value" => WavefunctionValue
    };

    let (wave_function, energies, errors) = {
        let sampler = Sampler::new(wave_function, metropolis::MetropolisDiffuse::from_rng(0.1, StdRng::from_seed([0_u8; 32])), &obs);

        // Construct the VMC runner, with Stochastic reconfiguration as optimizer
        // and an empty Logger so no output is given during each VMC iteration
        let vmc_runner = VmcRunner::new(
            sampler,
            //OnlineLbfgs::new(0.001, 5, NPARM_JAS),
            //NesterovMomentum::new(0.01, 0.00001, NPARM_JAS),
            //SteepestDescent::new(0.0001),
            StochasticReconfiguration::new(100.0),
            EmptyLogger { block_size: BLOCK_SIZE },
        );

        // Actually run the VMC optimization
        vmc_runner.run_optimization(NITERS, TOTAL_SAMPLES, BLOCK_SIZE, NWORKERS)
    };

    // Plot the results
    plot_results(energies.as_slice().unwrap(), errors.as_slice().unwrap());
}

fn plot_results(energy: &[f64], error: &[f64]) {
    let niters = energy.len();
    let iters: Vec<_> = (0..niters).collect();
    let exact = vec![-2.903; niters];

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
