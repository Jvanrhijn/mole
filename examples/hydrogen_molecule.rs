use std::collections::HashMap;

#[macro_use]
extern crate itertools;
use gnuplot::{AxesCommon, Caption, Color, Figure, FillAlpha};
use ndarray::{array, s, Array1, Array2, Ix2};
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
        let _energy = data
            .get("Energy")
            .unwrap()
            .chunks(self.block_size)
            .last()
            .unwrap()
            .iter()
            .fold(0.0, |a, b| a + b.get_scalar().unwrap())
            / self.block_size as f64;
        //format!("\tBlock energy:    {:.8}", energy)
        String::new()
    }
}

#[derive(Clone)]
struct STO {
    alpha: f64,
}

impl STO {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn value(&self, x: &Array1<f64>) -> f64 {
        f64::exp(-self.alpha * x.norm_l2())
    }

    pub fn gradient(&self, x: &Array1<f64>) -> Array1<f64> {
        -self.alpha * self.value(x) / x.norm_l2() * x
    }

    pub fn laplacian(&self, x: &Array1<f64>) -> f64 {
        self.alpha * self.value(x) / x.norm_l2() * (self.alpha * x.norm_l2() - 2.0)
    }

    pub fn parameter_gradient(&self, x: &Array1<f64>) -> f64 {
        -x.norm_l2() * self.value(x)
    }

    pub fn update_parameters(&mut self, deltap: f64) {
        self.alpha += deltap;
    }
}

#[derive(Clone)]
struct HydrogenMoleculeWaveFunction {
    nuclear_separation: f64,
    params: Array1<f64>,
    phi: STO,
}

impl HydrogenMoleculeWaveFunction {
    pub fn new(nuclear_separation: f64, params: Array1<f64>) -> Self {
        let alpha = params[0];
        Self {
            nuclear_separation,
            params,
            phi: STO::new(alpha),
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
        let r = Array1::<f64>::from_vec(vec![self.nuclear_separation, 0.0, 0.0]);
        Ok(
            self.phi.value(&(&x1 - &(0.5 * &r))) * self.phi.value(&(&x2 + &(0.5 * &r)))
                + self.phi.value(&(&x1 + &(0.5 * &r))) * self.phi.value(&(&x2 - &(0.5 * &r))),
        )
    }
}

impl Differentiate for HydrogenMoleculeWaveFunction {
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Result<Array2<f64>> {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        // parameters
        let r = Array1::<f64>::from_vec(vec![self.nuclear_separation, 0.0, 0.0]);
        //
        let mut grad = Array2::<f64>::zeros((2, 3));
        let mut gradx1 = grad.slice_mut(s![0, ..]);
        gradx1 += &(self.phi.value(&(&x2 + &(0.5 * &r))) * self.phi.gradient(&(&x1 - &(0.5 * &r)))
            + self.phi.value(&(&x2 - &(0.5 * &r))) * self.phi.gradient(&(&x1 + &(0.5 * &r))));

        let mut gradx2 = grad.slice_mut(s![1, ..]);
        gradx2 += &(self.phi.value(&(&x1 + &(0.5 * &r))) * self.phi.gradient(&(&x2 - &(0.5 * &r)))
            + self.phi.value(&(&x1 - &(0.5 * &r))) * self.phi.gradient(&(&x2 + &(0.5 * &r))));

        Ok(grad)
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64> {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        // parameters
        let r = Array1::<f64>::from_vec(vec![self.nuclear_separation, 0.0, 0.0]);
        let laplpsi = &(self.phi.value(&(&x2 + &(0.5 * &r)))
            * self.phi.laplacian(&(&x1 - &(0.5 * &r)))
            + self.phi.value(&(&x2 - &(0.5 * &r))) * self.phi.laplacian(&(&x1 + &(0.5 * &r))))
            + &(self.phi.value(&(&x1 + &(0.5 * &r))) * self.phi.laplacian(&(&x2 - &(0.5 * &r)))
                + self.phi.value(&(&x1 - &(0.5 * &r))) * self.phi.laplacian(&(&x2 + &(0.5 * &r))));
        Ok(laplpsi)
    }
}

impl Optimize for HydrogenMoleculeWaveFunction {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>> {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        // parameters
        let r = Array1::<f64>::from_vec(vec![self.nuclear_separation, 0.0, 0.0]);
        Ok(array![
            self.phi.value(&(&x1 - &(0.5 * &r)))
                * self.phi.parameter_gradient(&(&x2 + &(0.5 * &r)))
                + self.phi.parameter_gradient(&(&x1 - &(0.5 * &r)))
                    * self.phi.value(&(&x2 + &(0.5 * &r)))
                + self.phi.value(&(&x1 + &(0.5 * &r)))
                    * self.phi.parameter_gradient(&(&x2 - &(0.5 * &r)))
                + self.phi.parameter_gradient(&(&x1 + &(0.5 * &r)))
                    * self.phi.value(&(&x2 - &(0.5 * &r)))
        ])
    }

    fn update_parameters(&mut self, deltap: &Array1<f64>) {
        self.params += deltap;
        self.phi.update_parameters(deltap[0]);
    }

    fn parameters(&self) -> &Array1<f64> {
        &self.params
    }

    fn num_parameters(&self) -> usize {
        self.params.len()
    }
}

// Number of VMC iterations
static NITERS: usize = 10;
// Number of threads
static NWORKERS: usize = 8;
// Total number of MC samples, distributed over workers
// samples per worker is TOTAL_SAMPLES / NWORKERS
static TOTAL_SAMPLES: usize = 20_000;
// Block size for blocking analysis. Effective number of
// samples, assuming unit correlation time:
// TOTAL_SAMPLES - BLOCK_SIZE * NWORKERS
static BLOCK_SIZE: usize = 10;

fn main() {
    // H2 equilibrium geometry
    let ion_pos = array![[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];
    let sep = ion_pos[[1, 0]] - ion_pos[[0, 0]];

    // construct  wave function
    let wave_function = HydrogenMoleculeWaveFunction::new(sep, array![0.5]);

    // run optimization for two different optimizers
    println!("STOCHASTIC RECONFIGURATION");
    let (sr_wf, energies_sr, errors_sr) = optimize_wave_function(
        &ion_pos,
        wave_function.clone(),
        StochasticReconfiguration::new(50_000.0),
    );
    println!("\nSTEEPEST DESCENT");
    let (sd_wf, energies_sd, errors_sd) =
        optimize_wave_function(&ion_pos, wave_function.clone(), SteepestDescent::new(1e-5));
    println!();

    const NUM_WALKERS: usize = 500;
    const TAU: f64 = 1e-2;
    const NUM_ITERS: usize = 20_000;
    const DMC_BLOCK_SIZE: usize = 100;

    let hamiltonian = ElectronicHamiltonian::from_ions(ion_pos.clone(), array![1, 1]);
    let metrop = MetropolisDiffuse::from_rng(TAU, StdRng::from_seed([0_u8; 32]));

    let mut dmc = DmcRunner::new(
        sr_wf,
        NUM_WALKERS,
        *energies_sr.to_vec().last().unwrap(),
        hamiltonian,
        metrop,
        SRBrancher::new(),
    );

    let (energies, errs) = dmc.diffuse(TAU, NUM_ITERS, DMC_BLOCK_SIZE);

    // Plot the results
    plot_results(
        &[energies_sr, energies_sd],
        &[errors_sr, errors_sd],
        &["blue", "red"],
        &["Stochastic Refonfiguration", "Steepest Descent"],
    );

    plot_results_dmc(
        &energies.into(),
        &errs.into(),
        "blue",
    );
}

fn optimize_wave_function<O: Optimizer + Send + Sync + Clone>(
    ion_pos: &Array2<f64>,
    wave_function: HydrogenMoleculeWaveFunction,
    opt: O,
) -> (HydrogenMoleculeWaveFunction, Array1<f64>, Array1<f64>) {
    //  hamiltonian operator
    let hamiltonian = ElectronicHamiltonian::from_ions(ion_pos.clone(), array![1, 1]);

    let obs = operators! {
        "Energy" => hamiltonian,
        "Parameter gradient" => ParameterGradient,
        "Wavefunction value" => WavefunctionValue
    };

    let (wave_function, energies, errors) = {
        let sampler = Sampler::new(
            wave_function,
            metropolis::MetropolisDiffuse::from_rng(0.25, StdRng::from_seed([0_u8; 32])),
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

    (wave_function, energies, errors)
}

fn plot_results(
    energies: &[Array1<f64>],
    errors: &[Array1<f64>],
    colors: &[&str],
    labels: &[&str],
) {
    let niters = energies[0].len();
    let iters: Vec<_> = (0..niters).collect();
    let exact = vec![-1.17447; niters];

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

fn plot_results_dmc(energies: &Array1<f64>, errors: &Array1<f64>, color: &str) {
    let niters = energies.len();
    let iters: Vec<_> = (0..niters).collect();
    let exact = vec![-1.17447; niters];

    let mut fig = Figure::new();
    let axes = fig.axes2d();
    axes.fill_between(
        &iters,
        &(energies - errors),
        &(energies + errors),
        &[Color(color), FillAlpha(0.1)],
    )
    .lines(&iters, energies, &[Caption("DMC energy"), Color(color)]);
    axes.lines(
        &iters,
        &exact,
        &[Caption("Best ground state energy, H2"), Color("black")],
    )
    .set_x_label("Iteration", &[])
    .set_y_label("Exact energy (Hartree)", &[])
    .set_x_grid(true)
    .set_y_grid(true);

    fig.show();
}
