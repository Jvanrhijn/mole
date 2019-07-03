use std::collections::HashMap;
#[macro_use]
extern crate ndarray;
use basis::Hydrogen1sBasis;
use metropolis::MetropolisDiffuse;
use montecarlo::{traits::{Log, MonteCarloSampler}, Runner, Sampler};
use ndarray::{Array1, Axis};
use operator::{
    ElectronicHamiltonian, ElectronicPotential, IonicPotential, KineticEnergy, OperatorValue,
};
use rand::{SeedableRng, StdRng};
use wavefunction::{Orbital, SpinDeterminantProduct};

struct MockLogger;
impl Log for MockLogger {
    fn log(&mut self, _data: &HashMap<String, Vec<OperatorValue>>) -> String {
        String::new()
    }
}

#[test]
fn helium_lcao() {
    let optimal_width = 1.0 / 1.69;

    let ion_pos = array![[0.0, 0.0, 0.0]];
    let basis = Hydrogen1sBasis::new(ion_pos.clone(), vec![optimal_width]);

    let orbitals = vec![
        Orbital::new(array![[1.0]], basis.clone()),
        Orbital::new(array![[1.0]], basis.clone()),
    ];

    let wave_function = SpinDeterminantProduct::new(orbitals, 1);

    let kinetic = KineticEnergy::new();
    let potential_ions = IonicPotential::new(ion_pos, array![2]);
    let potential_elec = ElectronicPotential::new();
    let hamiltonian = ElectronicHamiltonian::new(kinetic, potential_ions, potential_elec);

    let rng = StdRng::from_seed([0u8; 32]);
    let metrop = MetropolisDiffuse::from_rng(0.1, rng);

    let mut sampler = Sampler::new(wave_function, metrop);
    sampler.add_observable("Energy", hamiltonian);

    let mut runner = Runner::new(sampler, MockLogger);
    let result = runner.run(1000, 100);

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
