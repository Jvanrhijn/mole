use mole::prelude::*;
use ndarray::{Array1, Array2, Array, Ix1, Ix2, Axis, array, s};
use ndarray_linalg::Norm;
use std::collections::HashMap;
use rand::{SeedableRng, StdRng};

use rand::distributions::{Normal, Uniform, WeightedChoice, Weighted, Distribution};
use rand::{FromEntropy, Rng, RngCore};
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

    let vmc_energy = energies.iter().last().unwrap();
    let error = errors.iter().last().unwrap();

    println!("\nVMC Energy:     {} +/- {:.*}\n", vmc_energy, 8, error);

    // now do a DMC run

    // first, extract the standard deviation
    // of the probability density |psi^2|
    let alpha = guiding_wf.parameters()[0];
    let stdev = alpha/2.0;

    // sample a set of starting configurations
    // from the wave function
    let num_confs = 16_000;
    const TAU: f64 = 0.001;
    const DMC_ITERS: usize = 25000;
    const EQ_ITERS: usize = DMC_ITERS / 10;

    let mut rng = StdRng::from_seed([1_u8; 32]);
    let mut confs = vec![];
    for _ in 0..num_confs {
        confs.push((1.0, Array2::random_using((1, 3), Normal::new(0.0, stdev), &mut rng)));
    }

    // initialize trial energy
    let mut trial_energy = *vmc_energy;
    // initialize dmc energy
    let mut dmc_energy = trial_energy;
    let mut dmc_energy_variance = 0.0;
    
    let mut metrop = MetropolisDiffuse::from_rng(TAU, rng.clone());


    // for the number of dmc iterations
    for j in 0..DMC_ITERS {
        let mut energy_acc = 0.0;

        // for each configuration
        for i in 0..confs.len() {
            // propose electron move using metropolis diffusion algorithm
            let (mut weight, conf) = &confs[i];
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
            let local_e_new = hamiltonian.act_on(&guiding_wf, &new_conf).unwrap().get_scalar().unwrap()/guiding_wf.value(&new_conf).unwrap();
        
            weight *= f64::exp(-TAU * (0.5*(local_e + local_e_new) - trial_energy));
            confs[i].0 = weight;
            confs[i].1 = new_conf;
            energy_acc += weight*local_e_new;
        }

        let global_weight = confs.iter().fold(0.0, |acc, (weight, _)| acc + *weight);

        energy_acc /= global_weight;

        // update trial energy
        trial_energy = (energy_acc  + trial_energy) / 2.0;

        // Perform stochastic reconfiguration
        let new_weight = global_weight / num_confs as f64;
        // construct list of weighed configurations
        let mut confs_weighted: Vec<_> = confs.iter()
                                  .map(|(w, c)| Weighted { weight: (w*100.0) as u32, item: c })
                                  .collect();
        let wc = WeightedChoice::new(&mut confs_weighted);
        // construct new configurations
        let mut new_confs = vec![];
        for _ in 0..num_confs {
            new_confs.push((new_weight, wc.sample(&mut rng).clone()))
        }

        confs = new_confs;

        // update DMC energy after equilibrating
        if j > EQ_ITERS {
            let dmc_energy_prev = dmc_energy;
            dmc_energy += (trial_energy - dmc_energy) / ((j - EQ_ITERS) + 1) as f64;
            dmc_energy_variance += ((trial_energy - dmc_energy_prev)*(trial_energy - dmc_energy) - dmc_energy_variance)
                /(j - EQ_ITERS + 1) as f64;
        }

        println!("Reference Energy:   {:.8}    DMC Energy:   {:.8} +/- {:.8}", trial_energy, dmc_energy, dmc_energy_variance.sqrt());
    }

    println!("DMC Energy:   {:.8} +/- {:.8}", dmc_energy, dmc_energy_variance.sqrt());
}
