use std::collections::HashMap;
use ndarray_linalg::Norm;
use metropolis::MetropolisBox;
use ndarray::{Array1, Axis, Array2, Ix2, array};
use operator::{
    ElectronicHamiltonian, ElectronicPotential, IonicPotential, KineticEnergy, OperatorValue,
};
use mole::prelude::*;
extern crate util;

use rand::{SeedableRng, StdRng};

struct MockLogger;
impl Log for MockLogger {
    fn log(&mut self, _data: &HashMap<String, Vec<OperatorValue>>) -> String {
        String::new()
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

    pub fn value(&self, x: &Array2<f64>) -> f64 {
        f64::exp(-self.alpha*x.norm_l2())
    }

    pub fn gradient(&self, x: &Array2<f64>) -> Array2<f64> {
        -self.alpha*self.value(x)/x.norm_l2() * x
    }

    pub fn laplacian(&self, x: &Array2<f64>) -> f64 {
        self.alpha*self.value(x)/x.norm_l2() * (self.alpha*x.norm_l2() - 2.0)
    }

    pub fn parameter_gradient(&self, x: &Array2<f64>) -> f64 {
        -x.norm_l2()*self.value(x)
    }

    pub fn update_parameters(&mut self, deltap: f64) {
        self.alpha += deltap;
    }
}

#[derive(Clone)]
struct H2WF {
    r: Array2<f64>,
    phi: STO,
}

impl H2WF {
    pub fn new(r: f64, alpha: f64) -> Self {
        Self { r: array![[r, 0.0, 0.0]], phi: STO::new(alpha) }
    }
}

impl Function<f64> for H2WF {
    type D = Ix2;

    fn value(&self, x: &Array2<f64>) -> Result<f64> {
        let r1 = &(x - &(0.5*&self.r));
        let r2 = &(x + &(0.5*&self.r));
        Ok(
            self.phi.value(&r1)*self.phi.value(&r2)
        )
    }
}

impl Differentiate for H2WF {
    type D = Ix2;

    fn gradient(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let r1 = &(x - &(0.5*&self.r));
        let r2 = &(x + &(0.5*&self.r));
        Ok(
            self.phi.value(r1)*self.phi.gradient(r2) + self.phi.value(r2)*self.phi.gradient(r1)
        )
    }

    fn laplacian(&self, x: &Array2<f64>) -> Result<f64> {
        let r1 = &(x - &(0.5*&self.r));
        let r2 = &(x + &(0.5*&self.r));
        Ok(
            self.phi.value(r1)*self.phi.laplacian(r2) + self.phi.value(r2)*self.phi.laplacian(r1)
            + 2.0*(&self.phi.gradient(r1) * &self.phi.gradient(r2)).sum()
        )
    }
}

impl WaveFunction for H2WF {
    fn num_electrons(&self) -> usize {
        1
    }
}

#[test]
fn hydrogen_molecular_ion_lcao() {
    let ion_pos = array![[-1.25, 0.0, 0.0], [1.25, 0.0, 0.0]];
    //let basis = Hydrogen1sBasis::new(ion_pos.clone(), vec![1.0]);

    //let orbitals = vec![Orbital::new(array![[1.0], [1.0]], basis.clone())];

    //let wave_function = SingleDeterminant::new(orbitals).unwrap();
    let wave_function = H2WF::new(2.5, 1.0);

    let kinetic = KineticEnergy::new();
    let potential_ions = IonicPotential::new(ion_pos, array![1, 1]);
    let potential_elec = ElectronicPotential::new();
    let hamiltonian = ElectronicHamiltonian::new(kinetic, potential_ions, potential_elec);

    let rng = StdRng::from_seed([0u8; 32]);
    let metrop = MetropolisBox::from_rng(1.0, rng);

    let obs = operators! {
        "Energy" => hamiltonian
    };

    let sampler = Sampler::new(wave_function, metrop, &obs).unwrap();

    let runner = Runner::new(sampler, MockLogger);
    let result = runner.run(10000, 100).unwrap();

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

    let exact_result = -0.565;
    assert!((energy - exact_result).abs() < energy_err);
}
