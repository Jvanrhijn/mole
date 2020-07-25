// This tests the optimization of a simple
// quantum harmonic oscillator wave function
use mole::prelude::*;
use ndarray::{array, Array1, Array2, Ix2};
use ndarray_linalg::Norm;

use rand::{SeedableRng, StdRng};
use std::collections::HashMap;

#[derive(Clone)]
struct EmptyLogger;
impl Log for EmptyLogger {
    fn log(&mut self, _data: &HashMap<String, Vec<OperatorValue>>) -> String {
        String::new()
    }
}

// implement SHO Hamiltonian + wave function
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
        Ok(f64::exp(-(x.norm_l2() / a).powi(2)))
    }
}

impl Differentiate for GaussianWaveFunction {
    type D = Ix2;

    fn gradient(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let a = self.params[0];
        Ok(-2.0 * self.value(x)? / a.powi(2) * x)
    }

    fn laplacian(&self, x: &Array2<f64>) -> Result<f64> {
        let a = self.params[0];
        Ok(-2.0*self.value(x)?*(a.powi(2) - 2.0*x.norm_l2().powi(2))/a.powi(4))
    }
}

impl WaveFunction for GaussianWaveFunction {
    fn num_electrons(&self) -> usize {
        1
    }
    
    fn dimension(&self) -> usize {
        1
    }
}

impl Optimize for GaussianWaveFunction {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>> {
        let a = self.params[0];
        Ok(array![
            self.value(cfg)? * 2.0 * cfg.norm_l2().powi(2) / a.powi(3)
        ])
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

static ITERS: usize = 20;
static SAMPLES: usize = 8000;
static BLOCK_SIZE: usize = 10;

#[test]
fn sho_optimize() {
    let wf = GaussianWaveFunction::new(1.0_f64.sqrt());
    let h = HarmonicHamiltonian::new(1.0);

    let metrop = MetropolisDiffuse::from_rng(0.1, StdRng::from_seed([0_u8; 32]));
    //let metrop = MetropolisBox::from_rng(0.25, StdRng::from_seed([0_u8; 32]));

    let obs = operators! {
        "Energy" => h,
        "Parameter gradient" => ParameterGradient,
        "Wavefunction value" => WavefunctionValue
    };

    let sampler = Sampler::with_initial_configuration(wf, metrop, &obs, array![[0.0]]).unwrap();

    let vmc = VmcRunner::new(sampler, StochasticReconfiguration::new(-1e7), EmptyLogger);

    let (_, energies, errors) = vmc.run_optimization(ITERS, SAMPLES, BLOCK_SIZE, 4).unwrap();

    let energy = energies.iter().last().unwrap();
    let error = *errors.iter().last().unwrap();

    assert!((energy - 0.5).abs() < error);
}
