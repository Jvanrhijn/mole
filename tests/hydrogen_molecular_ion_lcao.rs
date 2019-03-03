#[macro_use]
extern crate ndarray;
use basis::Hydrogen1sBasis;
use wavefunction::{SingleDeterminant, Orbital};
use operator::{ElectronicHamiltonian, IonicPotential, KineticEnergy, ElectronicPotential};
use metropolis::MetropolisBox;
use montecarlo::{Sampler, Runner};

use rand::{SeedableRng, StdRng};

#[test]
fn hydrogen_molecular_ion_lcao() {
    let ion_pos = array![[-1.2, 0.0, 0.0], [1.2, 0.0, 0.0]];
    let basis = Hydrogen1sBasis::new(ion_pos.clone(), vec![1.0]);

    let orbitals = vec![
        Orbital::new(array![[1.0], [1.0]], basis.clone()),
    ];

    let wave_function = SingleDeterminant::new(orbitals);

    let kinetic = KineticEnergy::new();
    let potential_ions = IonicPotential::new(ion_pos, array![1, 1]);
    let potential_elec = ElectronicPotential::new();
    let hamiltonian = ElectronicHamiltonian::new(kinetic, potential_ions, potential_elec);

    let rng = StdRng::from_seed([0u8; 32]);
    let metrop = MetropolisBox::from_rng(1.0, rng);

    let mut sampler = Sampler::new(wave_function, metrop);
    sampler.add_observable("Energy", hamiltonian);

    let mut runner = Runner::new(sampler);
    runner.run(1000, 100);

    let energy = *runner.means().get("Energy").unwrap();
    let energy_err = *runner.errors().get("Energy").unwrap();

    let exact_result = -0.56617;
    assert!((energy - exact_result).abs() < 2.0*energy_err);

}