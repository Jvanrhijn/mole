use std::collections::HashMap;

#[macro_use]
extern crate itertools;
use gnuplot::{AxesCommon, Caption, Color, Figure, FillAlpha};
use ndarray::{array, s, Array, Array1, Array2, Ix2};
use ndarray_linalg::Norm;
use rand::{SeedableRng, StdRng};

use mole::optimize::Optimize;
use mole::prelude::*;

#[derive(Clone)]
struct EmptyLogger {
    block_size: usize,
}
impl Log for EmptyLogger {
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

// implement a custom wave function for this atom
#[derive(Clone)]
struct HeliumAtomWaveFunction {
    params: Array1<f64>,
}

impl HeliumAtomWaveFunction {
    pub fn new(alpha: f64) -> Self {
        Self {
            params: Array1::from_vec(vec![alpha]),
        }
    }

    fn extract_config(cfg: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        (
            cfg.slice(s![0, ..]).to_owned(),
            cfg.slice(s![1, ..]).to_owned(),
        )
    }
}

impl WaveFunction for HeliumAtomWaveFunction {
    fn num_electrons(&self) -> usize {
        2
    }
}

impl Function<f64> for HeliumAtomWaveFunction {
    type D = Ix2;

    fn value(&self, cfg: &Array<f64, Self::D>) -> Result<f64> {
        // as ansatz, we use a product of STOs
        // with adjustable exponents:
        // $\psi(x_1, x_1) = \exp(-\alpha |x_1| - \beta |x_2|)$
        let alpha = self.params[0];
        let (x1, x2) = Self::extract_config(cfg);
        Ok(f64::exp(-alpha * (x1.norm_l2() + x2.norm_l2())))
    }
}

impl Differentiate for HeliumAtomWaveFunction {
    type D = Ix2;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Result<Array2<f64>> {
        let alpha = self.params[0];
        let (x1, x2) = Self::extract_config(cfg);
        let value = self.value(cfg)?;
        let grad_x1 = -alpha / x1.norm_l2() * value * &x1;
        let grad_x2 = -alpha / x2.norm_l2() * value * &x2;
        let mut out = Array2::<f64>::zeros(cfg.dim());
        let mut first_comp = out.slice_mut(s![0, ..]);
        first_comp += &grad_x1;
        let mut second_comp = out.slice_mut(s![1, ..]);
        second_comp += &grad_x2;
        Ok(out)
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64> {
        let alpha = self.params[0];
        let (x1, x2) = Self::extract_config(cfg);
        let val = self.value(cfg)?;
        let x1norm = x1.norm_l2();
        let x2norm = x2.norm_l2();
        Ok(alpha / x1norm * val * (alpha * x1norm - 2.0)
            + alpha / x2norm * val * (alpha * x2norm - 2.0))
    }
}

impl Optimize for HeliumAtomWaveFunction {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>> {
        let (x1, x2) = Self::extract_config(cfg);
        Ok(-self.value(cfg)? * Array1::from_vec(vec![x1.norm_l2() + x2.norm_l2()]))
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

static NITERS: usize = 10;
static NWORKERS: usize = 8;
static TOTAL_SAMPLES: usize = 10_000;
static BLOCK_SIZE: usize = 10;

fn main() {
    let ion_pos = array![[0.0, 0.0, 0.0]];

    let wave_function = HeliumAtomWaveFunction::new(0.5);

    // run optimization
    println!("STOCHASTIC RECONFIGURATION");
    let (energies_sr, errors_sr) = optimize_wave_function(
        &ion_pos,
        wave_function.clone(),
        StochasticReconfiguration::new(100_000.0),
    );
    println!("\nSTEEPEST DESCENT");
    let (energies_sd, errors_sd) =
        optimize_wave_function(&ion_pos, wave_function.clone(), SteepestDescent::new(1e-5));

    // Plot the results
    plot_results(
        &[energies_sr, energies_sd],
        &[errors_sr, errors_sd],
        &["blue", "red"],
        &["SR", "SD"],
    );
}

fn optimize_wave_function<O: Optimizer + Send + Sync + Clone>(
    ion_pos: &Array2<f64>,
    wave_function: HeliumAtomWaveFunction,
    opt: O,
) -> (Array1<f64>, Array1<f64>) {
    //  hamiltonian operator
    let hamiltonian = ElectronicHamiltonian::from_ions(ion_pos.clone(), array![2]);

    let obs = operators! {
        "Energy" => hamiltonian,
        "Parameter gradient" => ParameterGradient,
        "Wavefunction value" => WavefunctionValue,
        "Kin. Energy" => KineticEnergy::new()
    };

    let (_wave_function, energies, errors) = {
        let sampler = Sampler::new(
            wave_function,
            metropolis::MetropolisDiffuse::from_rng(0.25, StdRng::from_seed([0_u8; 32])),
            //metropolis::MetropolisBox::from_rng(0.5, StdRng::from_seed([0_u8; 32])),
            &obs,
        )
        .expect("Bad initial configuration");

        // Construct the VMC runner, with Stochastic reconfiguration as optimizer
        // and an empty Logger so no output is given during each VMC iteration
        let vmc_runner = VmcRunner::new(
            sampler,
            opt,
            EmptyLogger {
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
    let exact = vec![-2.903; niters];

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
        &[Caption("Best ground state energy, Helium"), Color("black")],
    )
    .set_x_label("Iteration", &[])
    .set_y_label("VMC Energy (Hartree)", &[])
    .set_x_grid(true)
    .set_y_grid(true);

    fig.show();
}
