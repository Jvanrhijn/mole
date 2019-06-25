use std::collections::HashMap;
#[macro_use]
extern crate ndarray;
use basis::Hydrogen1sBasis;
use metropolis::MetropolisBox;
use montecarlo::{traits::Log, Runner, Sampler};
use ndarray::{Array1, Axis};
use operator::{
    ElectronicHamiltonian, ElectronicPotential, IonicPotential, KineticEnergy, OperatorValue,
};
use wavefunction::{Orbital, SingleDeterminant};

use rand::{SeedableRng, StdRng};

struct MockLogger;
impl Log for MockLogger {
    fn log(&mut self, data: &HashMap<String, Vec<OperatorValue>>) -> String {
        String::new()
    }
}

#[test]
fn hydrogen_molecular_ion_lcao() {
    let ion_pos = array![[-1.25, 0.0, 0.0], [1.25, 0.0, 0.0]];
    let basis = Hydrogen1sBasis::new(ion_pos.clone(), vec![1.0]);

    let orbitals = vec![Orbital::new(array![[1.0], [1.0]], basis.clone())];

    let wave_function = SingleDeterminant::new(orbitals);

    let kinetic = KineticEnergy::new();
    let potential_ions = IonicPotential::new(ion_pos, array![1, 1]);
    let potential_elec = ElectronicPotential::new();
    let hamiltonian = ElectronicHamiltonian::new(kinetic, potential_ions, potential_elec);

    let rng = StdRng::from_seed([0u8; 32]);
    let metrop = MetropolisBox::from_rng(1.0, rng);

    let mut sampler = Sampler::new(wave_function, metrop);
    sampler.add_observable("Energy", hamiltonian);

    let mut runner = Runner::new(sampler, MockLogger);
    runner.run(10000, 100);

    let energy_data = Array1::<f64>::from_vec(
        runner
            .data()
            .get("Energy")
            .unwrap()
            .iter()
            .map(|x| *x.get_scalar().unwrap())
            .collect::<Vec<_>>(),
    );

    let energy = *energy_data.mean_axis(Axis(0)).first().unwrap();
    let energy_err = *energy_data.std_axis(Axis(0), 0.0).first().unwrap();
    //let energy = runner.data().get("Energy").unwrap()
    //    .clone().into_iter().sum::<OperatorValue>() / OperatorValue::Scalar((10000 * 100) as f64);
    //let energy_err = *runner.errors().get("Energy").unwrap();

    let exact_result = -0.565;
    assert!((energy - exact_result).abs() < energy_err);
}
