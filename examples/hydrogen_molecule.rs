use std::collections::HashMap;

#[macro_use]
extern crate itertools;
use gnuplot::{AxesCommon, Caption, Color, Figure, FillAlpha};
use ndarray::{array, Array1, Array2, Ix2, s};
use ndarray_linalg::Norm;
use rand::{SeedableRng, StdRng};

#[allow(unused_imports)]
use mole::prelude::*;

#[derive(Clone)]
struct Logger {
    block_size: usize,
}
impl Log for Logger {
    fn log(&mut self, data: &HashMap<String, Vec<OperatorValue>>) -> String {
        let energy = data
            .get("Energy")
            .unwrap()
            .chunks(self.block_size)
            .last()
            .unwrap()
            .iter()
            .fold(0.0, |a, b| a + b.get_scalar().unwrap())
            / self.block_size as f64;
        format!("\tBlock energy:    {:.8}", energy)
    }
}

#[derive(Clone)]
struct HydrogenMoleculeWaveFunction {
    nuclear_separation: f64,
    params: Array1<f64>,
}

impl HydrogenMoleculeWaveFunction {
    pub fn new(nuclear_separation: f64, params: Array1<f64>) -> Self {
        Self {
            nuclear_separation,
            params
        }
    }
}

impl WaveFunction for HydrogenMoleculeWaveFunction {
    fn num_electrons(&self) -> usize {
        2
    }
}

impl Function<f64> for HydrogenMoleculeWaveFunction {
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64> {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        // parameters
        let c1 = self.params[0];
        let c2 = self.params[1];
        let alpha = self.params[2];
        let r = Array1::<f64>::from_vec(vec![self.nuclear_separation, 0.0, 0.0]);
        Ok(c1 * f64::exp(-alpha*(&x1 - &(0.5*&r)).norm_l2()) 
         + c2 * f64::exp(-alpha*(&x2 + &(0.5*&r)).norm_l2()))
    }
}

impl Differentiate for HydrogenMoleculeWaveFunction {
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Result<Array2<f64>> {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        // parameters
        let c1 = self.params[0];
        let c2 = self.params[1];
        let alpha = self.params[2];
        let r = Array1::<f64>::from_vec(vec![self.nuclear_separation, 0.0, 0.0]);
        let sep1 = (&x1 - &(0.5*&r)).norm_l2();
        let sep2 = (&x2 + &(0.5*&r)).norm_l2();
        //
        let mut grad = Array2::<f64>::zeros((2, 3));
        let mut gradx1 = grad.slice_mut(s![0, ..]);
        gradx1 += &(-alpha*c1/sep1*f64::exp(-alpha*sep1)*(&x1 - &(0.5*&r)));
        let mut gradx2 = grad.slice_mut(s![1, ..]);
        gradx2 += &(-alpha*c2/sep2*f64::exp(-alpha*sep2)*(&x2 + &(0.5*&r)));
        Ok(grad)
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64> {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        // parameters
        let c1 = self.params[0];
        let c2 = self.params[1];
        let alpha = self.params[2];
        let r = Array1::<f64>::from_vec(vec![self.nuclear_separation, 0.0, 0.0]);
        let sep1 = (&x1 - &(0.5*&r)).norm_l2();
        let sep2 = (&x2 + &(0.5*&r)).norm_l2();
        Ok(
            alpha*c1*f64::exp(-alpha*sep1)/sep1*(alpha*sep1 - 1.0)
            + alpha*c2*f64::exp(-alpha*sep2)/sep2*(alpha*sep2 - 1.0)
        )
    }
}

impl Optimize for HydrogenMoleculeWaveFunction {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>> {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        // parameters
        let c1 = self.params[0];
        let c2 = self.params[1];
        let alpha = self.params[2];
        let r = Array1::<f64>::from_vec(vec![self.nuclear_separation, 0.0, 0.0]);
        let sep1 = (&x1 - &(0.5*&r)).norm_l2();
        let sep2 = (&x2 + &(0.5*&r)).norm_l2();
        Ok(Array1::<f64>::from_vec(vec![
           f64::exp(-alpha*sep1),
           f64::exp(-alpha*sep2), 
            -(c1 * sep1 * f64::exp(-alpha*sep1) + c2*sep2 * f64::exp(-alpha*sep2)),
        ]))
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

// Number of VMC iterations
static NITERS: usize = 20;
// Number of threads
static NWORKERS: usize = 8;
// Total number of MC samples, distributed over workers
// samples per worker is TOTAL_SAMPLES / NWORKERS
static TOTAL_SAMPLES: usize = 5000;
// Block size for blocking analysis. Effective number of
// samples, assuming unit correlation time:
// TOTAL_SAMPLES - BLOCK_SIZE * NWORKERS
static BLOCK_SIZE: usize = 50;


fn main() {
    // H2 equilibrium geometry
    let ion_pos = array![[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];
    let sep = ion_pos[[0, 1]] - ion_pos[[0, 2]];

    // construct  wave function
    let wave_function = HydrogenMoleculeWaveFunction::new(sep, vec![1.0, 1.0, 0.5].into());

    // run optimization for two different optimizers
    let (energies_sr, errors_sr) = optimize_wave_function(
        &ion_pos,
        wave_function.clone(),
        StochasticReconfiguration::new(1.0),
    );
    let (energies_sd, errors_sd) =
        optimize_wave_function(&ion_pos, wave_function.clone(), SteepestDescent::new(0.05));

    // Plot the results
    plot_results(
        &[energies_sr, energies_sd],
        &[errors_sr, errors_sd],
        &["blue", "red"],
        &["Stochastic Refonfiguration", "Steepest Descent"],
    );
}

fn optimize_wave_function<O: Optimizer + Send + Sync + Clone>(
    ion_pos: &Array2<f64>,
    wave_function: HydrogenMoleculeWaveFunction,
    opt: O,
) -> (Array1<f64>, Array1<f64>) {
    //  hamiltonian operator
    let hamiltonian = ElectronicHamiltonian::from_ions(ion_pos.clone(), array![1, 1]);

    let obs = operators! {
        "Energy" => hamiltonian,
        "Parameter gradient" => ParameterGradient,
        "Wavefunction value" => WavefunctionValue
    };

    let (_wave_function, energies, errors) = {
        let sampler = Sampler::new(
            wave_function,
            metropolis::MetropolisBox::from_rng(0.25, StdRng::from_seed([0_u8; 32])),
            &obs,
        )
        .expect("Bad initial configuration");

        // Construct the VMC runner, with Stochastic reconfiguration as optimizer
        // and an empty Logger so no output is given during each VMC iteration
        let vmc_runner = VmcRunner::new(
            sampler,
            opt,
            Logger {
                block_size: BLOCK_SIZE,
            },
        );

        // Actually run the VMC optimization
        vmc_runner.run_optimization(NITERS, TOTAL_SAMPLES, BLOCK_SIZE, NWORKERS)
    }
    .expect("VMC optimization failed");

    (energies, errors)
}

fn plot_results(
    energies: &[Array1<f64>],
    errors: &[Array1<f64>],
    colors: &[&str],
    labels: &[&str],
) {
    let niters = energies[0].len();
    let iters: Vec<_> = (0..niters).collect();
    let exact = vec![-1.175; niters];

    let mut fig = Figure::new();
    let axes = fig.axes2d();
    for (energy, error, color, label) in izip!(energies, errors, colors, labels) {
        axes.fill_between(
            &iters,
            &(energy - error),
            &(energy + error),
            &[Color(color), FillAlpha(0.1)],
        )
        .lines(&iters, energy, &[Caption(label), Color(color)]);
    }
    axes.lines(
        &iters,
        &exact,
        &[Caption("Best ground state energy, H2"), Color("black")],
    )
    .set_x_label("Iteration", &[])
    .set_y_label("VMC Energy (Hartree)", &[])
    .set_x_grid(true)
    .set_y_grid(true);

    fig.show();
}
