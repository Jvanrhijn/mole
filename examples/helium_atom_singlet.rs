use std::collections::{HashMap, VecDeque};

#[macro_use]
extern crate itertools;
use gnuplot::{AxesCommon, Caption, Color, Figure, FillAlpha};
use ndarray::{array, Array1, Array2, Array, Ix2, s, stack, Axis};
use ndarray_linalg::Norm;
use rand::{SeedableRng, StdRng};

use mole::prelude::*;
use mole::optimize::Optimize;
use mole::errors::Error::EmptyCacheError;

type Vgl = (f64, Array2<f64>, f64);
type Ovgl = (Option<f64>, Option<Array2<f64>>, Option<f64>);

#[derive(Clone)]
struct EmptyLogger {
    block_size: usize,
}
impl Log for EmptyLogger {
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

// implement a custom wave function for this atom
#[derive(Clone)]
struct HeliumAtomWaveFunction {
    params: Array1<f64>,
    current_value_queue: VecDeque<f64>,
    current_grad_queue: VecDeque<Array2<f64>>,
    current_laplac_queue: VecDeque<f64>,
}

impl HeliumAtomWaveFunction {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            params: Array1::from_vec(vec![alpha, beta]),
            current_value_queue: VecDeque::from(vec![0.0]),
            current_grad_queue: VecDeque::from(vec![Array2::zeros((2, 3))]),
            current_laplac_queue: VecDeque::from(vec![0.0]),
        }
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
        let (alpha, beta) = (self.params[0], self.params[1]);
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        Ok(f64::exp(-alpha*x1.norm_l2() -beta*x2.norm_l2()))
    }
}

impl Differentiate for HeliumAtomWaveFunction {
    type D = Ix2;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Result<Array2<f64>> {
        let (alpha, beta) = (self.params[0], self.params[1]);
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        let value = self.value(cfg)?;
        let grad_x1 = -alpha/x1.norm_l2() * value * &x1;
        let grad_x2 = -beta/x2.norm_l2()* value * &x2;
        let mut out = Array2::<f64>::zeros(cfg.dim());
        let mut first_comp = out.slice_mut(s![0, ..]);
        first_comp += &grad_x1;
        let mut second_comp = out.slice_mut(s![1, ..]);
        second_comp += &grad_x2;
        Ok(out)
    }

    fn laplacian(&self, cfg:&Array<f64, Self::D>) -> Result<f64> {
        let (alpha, beta) = (self.params[0], self.params[1]);
        let x1 = cfg.slice(s![0, ..]).to_owned();
        let x2 = cfg.slice(s![1, ..]).to_owned();
        let grad = self.gradient(cfg)?;
        let grad_x1 = grad.slice(s![0, ..]);
        let grad_x2 = grad.slice(s![1, ..]);
        let val = self.value(cfg)?;
        let x1norm = x1.norm_l2();
        let x2norm = x2.norm_l2();
        Ok(
            alpha/x1norm * val * (alpha*x1norm - 2.0)
            + beta/x2norm * val * (beta*x2norm - 2.0)
        )
    }
}

impl Cache for HeliumAtomWaveFunction {
    type U = usize;

    /// Refresh the cached data
    fn refresh(&mut self, new: &Array2<f64>) -> Result<()> {
        *self
            .current_value_queue
            .front_mut()
            .ok_or(EmptyCacheError)? = self.value(new)?;
        *self
            .current_grad_queue
            .front_mut()
            .ok_or(EmptyCacheError)? = self.gradient(new)?;
        *self.current_laplac_queue
            .front_mut()
            .ok_or(EmptyCacheError)? = self.laplacian(new)?;
        Ok(()) 
    }

    /// Calculate updated value of the cache given update data and new configuration,
    /// and set this data enqueued
    fn enqueue_update(&mut self, ud: Self::U, new: &Array2<f64>) -> Result<()> {
        self.current_value_queue.push_back(self.value(new)?);
        self.current_grad_queue.push_back(self.gradient(new)?);
        self.current_laplac_queue.push_back(self.laplacian(new)?);
        Ok(())
    }

    /// Push enqueued update into cache
    fn push_update(&mut self) {
        for q in [
            &mut self.current_value_queue,
            &mut self.current_laplac_queue,
        ]
        .iter_mut()
        {
            if q.len() == 2 {
                q.pop_front();
            }
        }
        if self.current_grad_queue.len() == 2 {
            self.current_grad_queue.pop_front();
        }
    }

    /// Flush the enqueued update data
    fn flush_update(&mut self) {
        for q in [
            &mut self.current_value_queue,
            &mut self.current_laplac_queue,
        ]
        .iter_mut()
        {
            if q.len() == 2 {
                q.pop_back();
            }
        }
        if self.current_grad_queue.len() == 2 {
            self.current_grad_queue.pop_back();
        }
    }

    /// Return the current value of the cached data
    fn current_value(&self) -> Result<Vgl> {
        Ok((
            *self.current_value_queue.front().ok_or(EmptyCacheError)?,
            self.current_grad_queue.front().ok_or(EmptyCacheError)?.clone(),
            *self.current_laplac_queue.front().ok_or(EmptyCacheError)?,
        ))
    }

    fn enqueued_value(&self) -> Ovgl {
        (
            self.current_value_queue.back().copied(),
            self.current_grad_queue.back().cloned(),
            self.current_laplac_queue.back().copied(),
        )
    }
}

impl Optimize for HeliumAtomWaveFunction {
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>> {
        let x1 = cfg.slice(s![0, ..]);
        let x2 = cfg.slice(s![1, ..]);
        Ok(self.value(cfg)? * Array1::from_vec(vec![
            -x1.norm_l2(),
            -x2.norm_l2()
        ]))
    }

    fn update_parameters(&mut self, deltap: &Array1<f64>) {
        dbg!(deltap);
        self.params += deltap;
    }

    fn parameters(&self) -> &Array1<f64> {
        &self.params
    }

    fn num_parameters(&self) -> usize {
        self.params.len()
    }

}


static NITERS: usize = 20;
static NWORKERS: usize = 8;
static TOTAL_SAMPLES: usize = 10_000;
static BLOCK_SIZE: usize = 50;
static NPARM_JAS: usize = 2;

fn main() {
    let width = 0.6;
    // setup basis set
    let ion_pos = array![[0.0, 0.0, 0.0]];

    let basis_set = Hydrogen1sBasis::new(ion_pos.clone(), vec![width]);

    // construct orbitals
    let orbitals = vec![
        Orbital::new(array![[1.0]], basis_set.clone()),
        Orbital::new(array![[1.0]], basis_set.clone()),
    ];

    // construct Jastrow-Slater wave function
    //let wave_function = JastrowSlater::new(
    //    Array1::zeros(NPARM_JAS), // Jastrow factor parameters
    //    orbitals.clone(),
    //    0.001, // scale distance
    //    1,     // number of electrons with spin up
    //)
    //.expect("Bad wave function");
    let wave_function = HeliumAtomWaveFunction::new(0.3, 0.1);

    // run optimization
    let (energies_sr, errors_sr) = optimize_wave_function(
        &ion_pos,
        wave_function.clone(),
        StochasticReconfiguration::new(2500.0),
    );
    let (energies_sd, errors_sd) =
        optimize_wave_function(
            &ion_pos, 
            wave_function.clone(), 
            SteepestDescent::new(1e-6),
            //NesterovMomentum::new(1e-4, 1e-4, 2)
            //OnlineLbfgs::new(-1.0, 10, 2)
    );

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
        "Wavefunction value" => WavefunctionValue
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
            //OnlineLbfgs::new(0.1, 10, NPARM_JAS),
            //NesterovMomentum::new(0.01, 0.00001, NPARM_JAS),
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
