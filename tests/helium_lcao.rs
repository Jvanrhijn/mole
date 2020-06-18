use std::collections::HashMap;
#[macro_use]
extern crate ndarray;
use metropolis::MetropolisDiffuse;
use ndarray::{Array, Array1, Axis, Array2, Ix2};
use ndarray_linalg::Norm;
use operator::{
    ElectronicHamiltonian, ElectronicPotential, IonicPotential, KineticEnergy, OperatorValue,
};
use rand::{SeedableRng, StdRng};
use mole::prelude::*;
#[macro_use]
extern crate util;

struct MockLogger;
impl Log for MockLogger {
    fn log(&mut self, _data: &HashMap<String, Vec<OperatorValue>>) -> String {
        String::new()
    }
}

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
        (cfg.slice(s![0, ..]).to_owned(), cfg.slice(s![1, ..]).to_owned())
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
        Ok(f64::exp(-alpha*(x1.norm_l2() + x2.norm_l2())))
    }
}

impl Differentiate for HeliumAtomWaveFunction {
    type D = Ix2;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Result<Array2<f64>> {
        let alpha = self.params[0];
        let (x1, x2) = Self::extract_config(cfg);
        let value = self.value(cfg)?;
        let grad_x1 = -alpha/x1.norm_l2() * value * &x1;
        let grad_x2 = -alpha/x2.norm_l2()* value * &x2;
        let mut out = Array2::<f64>::zeros(cfg.dim());
        let mut first_comp = out.slice_mut(s![0, ..]);
        first_comp += &grad_x1;
        let mut second_comp = out.slice_mut(s![1, ..]);
        second_comp += &grad_x2;
        Ok(out)
    }

    fn laplacian(&self, cfg:&Array<f64, Self::D>) -> Result<f64> {
        let alpha = self.params[0];
        let (x1, x2) = Self::extract_config(cfg);
        let val = self.value(cfg)?;
        let x1norm = x1.norm_l2();
        let x2norm = x2.norm_l2();
        Ok(
            alpha/x1norm * val * (alpha*x1norm - 2.0)
            + alpha/x2norm * val * (alpha*x2norm - 2.0)
        )
    }
}

#[test]
fn helium_lcao() {
    let optimal_width = 1.0 / 1.69;

    let ion_pos = array![[0.0, 0.0, 0.0]];
    //let basis = Hydrogen1sBasis::new(ion_pos.clone(), vec![optimal_width]);

    //let orbitals = vec![
    //    Orbital::new(array![[1.0]], basis.clone()),
    //    Orbital::new(array![[1.0]], basis.clone()),
    //];

    //let wave_function = SpinDeterminantProduct::new(orbitals, 1).unwrap();
    let wave_function = HeliumAtomWaveFunction::new(1.0/optimal_width);

    let kinetic = KineticEnergy::new();
    let potential_ions = IonicPotential::new(ion_pos, array![2]);
    let potential_elec = ElectronicPotential::new();
    let hamiltonian = ElectronicHamiltonian::new(kinetic, potential_ions, potential_elec);

    let rng = StdRng::from_seed([0u8; 32]);
    let metrop = MetropolisDiffuse::from_rng(0.1, rng);

    let obs = operators! {
        "Energy" => hamiltonian
    };

    let sampler = Sampler::new(wave_function, metrop, &obs).unwrap();

    let runner = Runner::new(sampler, MockLogger);
    let result = runner.run(1000, 100).unwrap();

    let energy_data = Array1::<f64>::from_vec(
        result
            .data
            .get("Energy")
            .unwrap()
            .iter()
            .map(|x| *x.get_scalar().unwrap())
            .collect::<Vec<_>>(),
    );

    let energy = *energy_data.mean_axis(Axis(0)).first().unwrap();
    let energy_err = *energy_data.std_axis(Axis(0), 0.0).first().unwrap();

    let exact_result = 0.5 * (1.5_f64).powi(6) * (-0.5);
    dbg!(energy);
    dbg!(exact_result);
    assert!((energy - exact_result).abs() < 2.0 * energy_err);
}
