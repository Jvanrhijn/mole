use mole::prelude::*;
use ndarray::{Array1, Array2, Array, Ix1, Ix2, Axis, array, s};
use ndarray_linalg::Norm;
use std::collections::HashMap;
use rand::{SeedableRng, StdRng};

use rand::distributions::{Normal, Range, Uniform};
use rand::{FromEntropy, Rng};
use ndarray_rand::RandomExt;

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
    let ansatz = GaussianWaveFunction::new(1.0);

    let metrop = MetropolisDiffuse::from_rng(0.1, StdRng::from_seed([0; 32]));

    // Construct our custom operator
    let hamiltonian = ElectronicHamiltonian::from_ions(array![[0.0, 0.0, 0.0]], array![1]);

    let obs = operators! {
        "Energy" => hamiltonian.clone(),
        "Parameter gradient" => ParameterGradient,
        "Wavefunction value" => WavefunctionValue
    };

    let sampler = Sampler::new(ansatz, metrop, &obs).expect("Bad initial configuration");

    // first do a VMC run to obtain a variationally optimized wave function
    let vmc = VmcRunner::new(sampler, SteepestDescent::new(1e-5), Logger);

    let (mut guiding_wf, energies, errors) = vmc.run_optimization(ITERS, TOTAL_SAMPLES, BLOCK_SIZE, 4)
                                            .expect("VMC run failed");

    let energy = energies.iter().last().unwrap();
    let error = errors.iter().last().unwrap();

    println!("\nVMC Energy:     {} +/- {:.*}\n", energy, 8, error);

    // now do a DMC run

    // first, extract the standard deviation
    // of the probability density |psi^2|
    let alpha = guiding_wf.parameters()[0];
    let stdev = alpha/2.0;

    // sample a set of starting configurations
    // from the wave function
    let num_confs = 10_000;
    const TAU: f64 = 0.05;
    const DMC_ITERS: usize = 100;

    //let mut confs = Array2::random_using((num_confs, 3), Normal::new(0.0, stdev), &mut StdRng::from_seed([1_u8; 32]));
    let mut rng = StdRng::from_seed([1_u8; 32]);
    let mut confs = vec![];
    for _ in 0..num_confs {
        confs.push(Array2::random_using((1, 3), Normal::new(0.0, stdev), &mut rng));
    }

    // initialize trial energy
    let mut trial_energy = *energy;
    // initialize dmc energy
    let mut metrop = MetropolisDiffuse::from_rng(TAU, rng.clone());

    // for the number of dmc iterations
    for j in 0..DMC_ITERS {
        let mut energy_acc = 0.0;
        let mut branch_acc = 0.0;

        let mut to_kill = vec![];
        let mut to_birth = vec![];

        // for each configuration
        for i in 0..confs.len() {
            // propose electron move using metropolis diffusion algorithm
            let conf = &confs[i];
            let mut new_conf = if let Some(x) = metrop.move_state(&mut guiding_wf, &conf, 0).unwrap() {
                x
            } else {
                conf.clone()
            };

            // apply FN approximation (not really needed here but i'll do it anyway)
            if guiding_wf.value(conf).unwrap().signum() != guiding_wf.value(&new_conf).unwrap().signum() {
                new_conf = conf.clone();
            }


            // branching factor
            let local_e = hamiltonian.act_on(&guiding_wf, &conf).unwrap().get_scalar().unwrap()/guiding_wf.value(&conf).unwrap();
            let local_e_new = hamiltonian.act_on(&guiding_wf, &new_conf).unwrap().get_scalar().unwrap()/guiding_wf.value(&conf).unwrap();
        
            let branch_factor = f64::exp(-TAU * (0.5*(local_e + local_e_new) - trial_energy));
            branch_acc += branch_factor;
            energy_acc += local_e_new;

            // clone configuration according to branching algorithm
            let num_copies = ((branch_factor + rng.gen::<f64>()) as usize).min(3);

            if num_copies > 0 {
                for _ in 1..num_copies {
                    to_birth.push(new_conf.clone());
                }
            } else {
                to_kill.push(i);
            }
        }
        // kill walkers
        for i in to_kill.iter().rev() {
            confs.remove(*i);
        }

        // copy walkers
        confs.extend(to_birth);

        energy_acc /= branch_acc;
        // update trial energy, taking birth-death into account
        //trial_energy = energy_acc;
        trial_energy = energy_acc + (1.0 - confs.len() as f64 / num_confs as f64)/TAU;
   
        // randomly delete a number of walkers
        //let excess = confs.len() - num_confs;
        //for _ in 0..excess {
        //    confs.remove(rng.gen_range(0, confs.len()));
        //}

        println!("DMC Energy:   {:.8}", trial_energy);
    }
}
