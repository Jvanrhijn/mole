
use std::collections::HashMap;


use metropolis::MetropolisBox;
use mole::prelude::*;
use ndarray::{Array1, Array2, Axis, Ix2};
use ndarray_linalg::Norm;
use operator::{KineticEnergy, LocalOperator, OperatorValue};
use rand::{SeedableRng, StdRng};
use util::operators;
use wavefunction_traits::{Differentiate, Function};

// Create a very basic logger
struct Logger;
impl Log for Logger {
    fn log(&mut self, data: &HashMap<String, Vec<OperatorValue>>) -> String {
        format!(
            "Energy: {}",
            data.get("Energy")
                .unwrap()
                .iter()
                .last()
                .unwrap()
                .get_scalar()
                .unwrap()
        )
    }
}

// Create a struct to hold Hamiltonian parameters
struct HarmonicHamiltonian {
    // Harmonic oscillator potential is parametrized by natural frequency
    frequency: f64,
    // Kinetic energy operator always has the same form
    t: KineticEnergy,
}

impl HarmonicHamiltonian {
    pub fn new(frequency: f64) -> Self {
        Self {
            t: KineticEnergy::new(),
            frequency,
        }
    }
}

// All observables must implement the Operator<T> trait
// T is the type parameter of the wave function.
impl<T> LocalOperator<T> for HarmonicHamiltonian
where
    T: Function<f64, D = Ix2> + Differentiate<D = Ix2>,
{
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue> {
        // Kinetic energy
        let ke = self.t.act_on(wf, cfg)?;
        // Potential energy: V = 0.5*m*omega^2*|x|^2
        let pe = OperatorValue::Scalar(
            0.5 * self.frequency.powi(2) * cfg.norm_l2().powi(2) * wf.value(cfg)?,
        );
        Ok(&ke + &pe)
    }
}

#[derive(Clone)]
struct GaussianWaveFunction {
    a: f64,
}

impl GaussianWaveFunction {
    pub fn new(a: f64) -> Self {
        Self { a }
    }
}

impl Function<f64> for GaussianWaveFunction {
    type D = Ix2;

    fn value(&self, x: &Array2<f64>) -> Result<f64> {
        Ok(f64::exp(-(x.norm_l2() / self.a).powi(2)))
    }
}

impl Differentiate for GaussianWaveFunction {
    type D = Ix2;

    fn gradient(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        Ok(-2.0 * self.value(x)? / self.a.powi(2) * x)
    }

    fn laplacian(&self, x: &Array2<f64>) -> Result<f64> {
        Ok(self.value(x)? * (4.0 * x.norm_l2().powi(2) - 6.0 * self.a.powi(2)) / self.a.powi(4))
    }
}

impl WaveFunction for GaussianWaveFunction {
    fn num_electrons(&self) -> usize {
        1
    }
}

fn main() {
    let omega = 1.0;

    // Build wave function
    let ansatz = GaussianWaveFunction::new(f64::sqrt(2.0 / omega));

    let metrop = MetropolisBox::from_rng(1.0, StdRng::from_seed([0; 32]));

    // Construct our custom operator
    let hamiltonian = HarmonicHamiltonian::new(omega);

    let obs = operators! {
        "Energy" => hamiltonian
    };

    let sampler = Sampler::new(ansatz, metrop, &obs).expect("Bad initial configuration");

    // Perform the MC integration
    let runner = Runner::new(sampler, Logger);
    let result = runner.run(1000, 1).unwrap();

    let energy_data = Array1::<f64>::from_vec(
        result
            .data
            .get("Energy")
            .unwrap()
            .iter()
            .map(|x| *x.get_scalar().unwrap())
            .collect::<Vec<_>>(),
    );

    // Retrieve mean values of energy over run
    let energy = *energy_data.mean_axis(Axis(0)).first().unwrap();
    let error = *energy_data.std_axis(Axis(0), 0.0).first().unwrap();

    assert!((energy - 1.5).abs() < 1e-15);
    assert!(error < 1e-15);

    println!("\nEnergy:     {} +/- {:.*}", energy, 8, error);
}
