use std::collections::HashMap;

use gnuplot::{AxesCommon, Caption, Color, Figure};
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate ndarray;
use basis::Hydrogen1sBasis;
use errors::Error;
use montecarlo::{
    Sampler,
    traits::Log,
};
use ndarray::{Array1, Array2};
use operator::Operator;
use operator::{ElectronicHamiltonian, OperatorValue};
use optimize::{Optimize, SteepestDescent, NesterovMomentum, StochasticReconfiguration, OnlineLbfgs};
use rand::{SeedableRng, StdRng};
use wavefunction::{JastrowSlater, Orbital};
use wavefunction_traits::Cache;
use vmc::VmcRunner;

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

    const BLOCK_SIZE: usize = 10;

    // construct Jastrow-Slater wave function
    let wave_function = JastrowSlater::new(
        Array1::zeros(NPARM_JAS), // Jastrow factor parameters
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
            //OnlineLbfgs::new(0.001, 5, NPARM_JAS),
            //NesterovMomentum::new(0.01, 0.00001, NPARM_JAS),
            //SteepestDescent::new(0.0001),
            StochasticReconfiguration::new(100.0),
            EmptyLogger,
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
