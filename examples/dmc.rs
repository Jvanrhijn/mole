use mole::prelude::*;
use ndarray::{Array1, Array2, Array, Ix1, Ix2, Axis, array};
use ndarray_linalg::Norm;
use std::collections::HashMap;
use rand::{SeedableRng, StdRng};

// DMC test for hydrogen atom,
// testing ground for library integration of
// DMC algorithm

// Create a very basic logger
#[derive(Clone)]
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
        );
        String::new()
    }
}

// hydrogen atom trial function gaussian#[derive(Clone)]
#[derive(Clone)]
struct GaussianWaveFunction {
    params: Array1<f64>,
}

impl GaussianWaveFunction {
    pub fn new(a: f64) -> Self {
        Self { params: array![a] }
    }
}

impl Function<f64> for GaussianWaveFunction {
    type D = Ix2;

    fn value(&self, x: &Array2<f64>) -> Result<f64> {
        let a = self.params[0];
        Ok(f64::exp(-(x.norm_l2()/a).powi(2)))
    }
}

impl Differentiate for GaussianWaveFunction {
    type D = Ix2;

    fn gradient(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let a = self.params[0];
        Ok(-2.0*self.value(x)?/a.powi(2)*x)
    }

    fn laplacian(&self, x: &Array2<f64>) -> Result<f64> {
        let a = self.params[0];
        Ok(
            self.value(x)?*(4.0*x.norm_l2().powi(2) - 6.0*a.powi(2))/a.powi(4)
        )
    }
}

impl WaveFunction for GaussianWaveFunction {
    fn num_electrons(&self) -> usize {
        1
    }
}

impl Optimize for GaussianWaveFunction {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>> {
        let a = self.params[0];
        Ok(array![self.value(cfg)?*2.0*cfg.norm_l2().powi(2)/a.powi(3)])
    }

    fn update_parameters(&mut self, deltap: &Array1<f64>) {
        self.params += deltap;
    }

    fn parameters(&self) -> &Array1<f64> {
        &self.params
    }

    fn num_parameters(&self) -> usize {
        self.params.len()
    }
}

static ITERS: usize = 100;
static TOTAL_SAMPLES: usize = 5000;
static BLOCK_SIZE: usize = 10;

fn main() {
    // Build wave function
    let ansatz = GaussianWaveFunction::new(0.5);

    let metrop = MetropolisDiffuse::from_rng(0.1, StdRng::from_seed([0; 32]));

    // Construct our custom operator
    let hamiltonian = ElectronicHamiltonian::from_ions(array![[0.0, 0.0, 0.0]], array![1]);

    let obs = operators! {
        "Energy" => hamiltonian,
        "Parameter gradient" => ParameterGradient,
        "Wavefunction value" => WavefunctionValue
    };

    let sampler = Sampler::new(ansatz, metrop, &obs).expect("Bad initial configuration");

    // first do a VMC run to obtain a variationally optimized wave function
    let vmc = VmcRunner::new(sampler, SteepestDescent::new(1e-5), Logger);

    let (guiding_wf, energies, errors) = vmc.run_optimization(ITERS, TOTAL_SAMPLES, BLOCK_SIZE, 4)
                                            .expect("VMC run failed");

    let energy = energies.iter().last().unwrap();
    let error = errors.iter().last().unwrap();

    println!("\nVMC Energy:     {} +/- {:.*}", energy, 8, error);

    // now do a DMC run
}
