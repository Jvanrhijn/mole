use std::collections::HashMap;

#[macro_use]
extern crate itertools;
use gnuplot::{AxesCommon, Caption, Color, Figure, FillAlpha};
use ndarray::{array, Array1, Array2, Ix2, s, Ix1, ArrayView};
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
struct STO {
    alpha: f64
}

impl STO {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn value(&self, x: &Array1<f64>) -> f64 {
        f64::exp(-self.alpha*x.norm_l2())
    }

    pub fn gradient(&self, x: &Array1<f64>) -> Array1<f64> {
        -self.alpha*self.value(x)/x.norm_l2() * x
    }

    pub fn laplacian(&self, x: &Array1<f64>) -> f64 {
        self.alpha*self.value(x)/x.norm_l2() * (self.alpha*x.norm_l2() - 2.0)
    }

    pub fn parameter_gradient(&self, x: &Array1<f64>) -> f64 {
        -x.norm_l2()*self.value(x)
    }

    pub fn update_parameters(&mut self, deltap: f64) {
        self.alpha += deltap;
    }
}

#[derive(Clone)]
struct Jastrow {
    b: f64,
}

impl Jastrow {
    pub fn new(b: f64) -> Self {
        Self { b }
    }

    pub fn uvalue(&self, r: f64) -> f64 {
        r / (2.0 * (1.0 + self.b*r))
    }

    pub fn uderiv(&self, r: f64) -> f64 {
        1.0 / (2.0*(self.b*r + 1.0).powi(2))
    }

    pub fn uderiv2(&self, r: f64) -> f64 {
        -self.b / (self.b*r + 1.0).powi(3)
    }

    pub fn value(&self, cfg: &Array2<f64>) -> f64 {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        let x12 = (&x1 - &x2).norm_l2();
        f64::exp(self.uvalue(x12))
    }

    pub fn gradient(&self, cfg: &Array2<f64>) -> Array2<f64> {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        let x12 = (&x1 - &x2).norm_l2();
        let mut out = Array2::<f64>::zeros((2, 3));
        let mut grad1 = out.slice_mut(s![0, ..]);
        grad1 += &(self.uderiv(x12) / x12 * &x1);
        let mut grad2 = out.slice_mut(s![1, ..]);
        grad2 += &(self.uderiv(x12) / x12 * &x2);
        self.value(cfg)*out
    }

    pub fn laplacian(&self, cfg: &Array2<f64>) -> f64 {
        let x1: ArrayView<_, Ix1> = cfg.slice(s![0, ..]);
        let x2: ArrayView<_, Ix1> = cfg.slice(s![1, ..]);
        let x12 = (&x1 - &x2).norm_l2();
        // first coordinate
        let lapl1 = self.uderiv(x12)*(3.0/x12 - x1.dot(&(&x1 - &x2))) + self.uderiv2(x12) * x1.norm_l2()/x12;
        let lapl2 = self.uderiv(x12)*(3.0/x12 - x2.dot(&(&x1 - &x2))) + self.uderiv2(x12) * x2.norm_l2()/x12;
        let laplu = lapl1 + lapl2;
        self.value(cfg) * laplu + self.gradient(cfg).norm_l2().powi(2) / self.value(cfg)
    }

    pub fn parameter_gradient(&self, cfg: &Array2<f64>) -> f64 {
        let x1: ArrayView<_, Ix1> = cfg.slice(s![0, ..]);
        let x2: ArrayView<_, Ix1> = cfg.slice(s![1, ..]);
        let x12 = (&x1 - &x2).norm_l2();
        self.value(cfg)*(-x12.powi(2)/(2.0*(1.0 + self.b*x12).powi(2)))
    }
}

#[derive(Clone)]
struct HydrogenMoleculeWaveFunction {
    nuclear_separation: f64,
    params: Array1<f64>,
    phi: STO,
    jastrow: Jastrow,
}

impl HydrogenMoleculeWaveFunction {
    pub fn new(nuclear_separation: f64, params: Array1<f64>) -> Self {
        let alpha = params[0];
        let b = params[1];
        Self {
            nuclear_separation,
            params,
            phi: STO::new(alpha),
            jastrow: Jastrow::new(b),
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
        Ok(self.jastrow.value(cfg)*(self.phi.value(&(&x1 - &(0.5*&r)))*self.phi.value(&(&x2 + &(0.5*&r)))
            + self.phi.value(&(&x1 + &(0.5*&r)))*self.phi.value(&(&x2 - &(0.5*&r)))))
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
        gradx1 += &(self.phi.value(&(&x2 + &(0.5*&r))) * self.phi.gradient(&(&x1 - &(0.5*&r)))
                + self.phi.value(&(&x2 - &(0.5*&r))) * self.phi.gradient(&(&x1 + &(0.5*&r))));

        let mut gradx2 = grad.slice_mut(s![1, ..]);
        gradx2 += &(self.phi.value(&(&x1 + &(0.5*&r))) * self.phi.gradient(&(&x2 - &(0.5*&r)))
                + self.phi.value(&(&x1 - &(0.5*&r))) * self.phi.gradient(&(&x2 + &(0.5*&r))));

        Ok(grad*self.jastrow.value(cfg) + self.value(cfg)?*self.jastrow.gradient(cfg))
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64> {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        // parameters
        let r = Array1::<f64>::from_vec(vec![self.nuclear_separation, 0.0, 0.0]);
        let laplpsi = &(self.phi.value(&(&x2 + &(0.5*&r))) * self.phi.laplacian(&(&x1 - &(0.5*&r)))
                + self.phi.value(&(&x2 - &(0.5*&r))) * self.phi.laplacian(&(&x1 + &(0.5*&r))))
                + &(self.phi.value(&(&x1 + &(0.5*&r))) * self.phi.laplacian(&(&x2 - &(0.5*&r)))
                + self.phi.value(&(&x1 - &(0.5*&r))) * self.phi.laplacian(&(&x2 + &(0.5*&r))));
        let laplj = self.jastrow.laplacian(cfg);
        let gradpsi = self.gradient(cfg)?;
        let gradj = self.jastrow.gradient(cfg);
        //dbg!((&gradpsi * &gradj).sum());
        Ok(
            laplpsi*self.jastrow.value(cfg) + self.value(cfg)?*laplj + 2.0*(&gradpsi * &gradj).sum()
        )
    }
}

impl Optimize for HydrogenMoleculeWaveFunction {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>> {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        // parameters
        let r = Array1::<f64>::from_vec(vec![self.nuclear_separation, 0.0, 0.0]);
        Ok(array![
            self.phi.value(&(&x1 - &(0.5*&r)))*self.phi.parameter_gradient(&(&x2 + &(0.5*&r)))
            + self.phi.parameter_gradient(&(&x1 - &(0.5*&r)))*self.phi.value(&(&x2 + &(0.5*&r)))
            + self.phi.value(&(&x1 + &(0.5*&r)))*self.phi.parameter_gradient(&(&x2 - &(0.5*&r)))
            + self.phi.parameter_gradient(&(&x1 + &(0.5*&r)))*self.phi.value(&(&x2 - &(0.5*&r))),
            self.jastrow.parameter_gradient(cfg)
        ]
        )
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
static TOTAL_SAMPLES: usize = 10_000;
// Block size for blocking analysis. Effective number of
// samples, assuming unit correlation time:
// TOTAL_SAMPLES - BLOCK_SIZE * NWORKERS
static BLOCK_SIZE: usize = 50;


fn main() {
    // H2 equilibrium geometry
    let ion_pos = array![[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];
    let sep = ion_pos[[1, 0]] - ion_pos[[0, 0]];

    // construct  wave function
    let wave_function = HydrogenMoleculeWaveFunction::new(sep, array![0.5, 0.8]);

    // run optimization for two different optimizers
    let (energies_sr, errors_sr) = optimize_wave_function(
        &ion_pos,
        wave_function.clone(),
        StochasticReconfiguration::new(1.0),
    );
    let (energies_sd, errors_sd) =
        optimize_wave_function(
            &ion_pos, 
            wave_function.clone(), 
            SteepestDescent::new(0.0),
    );

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
            metropolis::MetropolisDiffuse::from_rng(1.0, StdRng::from_seed([0_u8; 32])),
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
