#[macro_use]
extern crate ndarray;
use basis::GaussianBasis;
use metropolis::MetropolisDiffuse;
use montecarlo::{Runner, Sampler};
use ndarray::{Array2, Ix2};
use ndarray_linalg::Norm;
use operator::{KineticEnergy, Operator};
use rand::{SeedableRng, StdRng};
use wavefunction::{Cache, Differentiate, Error, Function, Orbital, SingleDeterminant};

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
impl<'a, T> Operator<T> for HarmonicHamiltonian
where
    T: Function<f64, D = Ix2>
        + Differentiate<D = Ix2>
        + Cache<Array2<f64>, V = (f64, Array2<f64>, f64)>
        + ?Sized,
{
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<f64, Error> {
        // Kinetic energy
        let ke = self.t.act_on(wf, cfg)?;
        // Potential energy: V = 0.5*m*omega^2*|x|^2
        let pe = 0.5 * self.frequency.powi(2) * cfg.norm_l2().powi(2) * wf.current_value().0;
        Ok(ke + pe)
    }
}

fn main() {
    // Exact ground state of Harmonic oscillator
    let basis = GaussianBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0]);
    // the rest is the same as with other operators

    // Build wave function
    let orbital = Orbital::new(array![[1.0]], basis);
    let ansatz = SingleDeterminant::new(vec![orbital]);

    let metrop = MetropolisDiffuse::from_rng(1.0, StdRng::from_seed([0; 32]));

    // Construct our custom operator
    let hamiltonian = HarmonicHamiltonian::new(1.0);

    let mut sampler = Sampler::new(ansatz, metrop);
    sampler.add_observable("Energy", hamiltonian);

    // Perform the MC integration
    let mut runner = Runner::new(sampler);
    runner.run(1000, 1);

    let energy = *runner.means().get("Energy").unwrap();
    let stdev = (*runner.variances().get("Energy").unwrap()).sqrt();

    println!("\nEnergy:     {} +/- {:.*}", energy, 8, stdev);
}
