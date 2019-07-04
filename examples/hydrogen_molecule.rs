use std::collections::HashMap;
#[macro_use]
extern crate ndarray;
use rand::{SeedableRng, StdRng};

use gnuplot::{AxesCommon, Caption, Color, Figure};

use ndarray::{Array1, Array2, Axis, Ix2, Zip};

use basis::Hydrogen1sBasis;
use montecarlo::{
    traits::Log,
    Sampler,
};
use operator::{ElectronicHamiltonian, Operator, OperatorValue};
use optimize::{
    MomentumDescent, NesterovMomentum, OnlineLbfgs, Optimize, Optimizer, SteepestDescent,
    StochasticReconfiguration,
};
use errors::Error;
use vmc::VmcRunner;

use wavefunction::{JastrowSlater, Orbital};
use wavefunction_traits::Cache;

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
