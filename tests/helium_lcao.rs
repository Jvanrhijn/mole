#[macro_use]
extern crate ndarray;
use basis::Hydrogen1sBasis;
use wavefunction::{SpinDeterminantProduct, Orbital};
use operator::{ElectronicHamiltonian, IonicPotential, KineticEnergy, ElectronicPotential};
use metropolis::MetropolisDiffuse;
use montecarlo::{Sampler, Runner};
use rand::{SeedableRng,  StdRng};


#[test]
fn helium_lcao() {
    let optimal_width = 1.0/1.69;

    let ion_pos = array![[0.0, 0.0, 0.0]];
    let basis = Hydrogen1sBasis::new(ion_pos.clone(), vec![optimal_width]);

    let orbitals = vec![
        Orbital::new(array![[1.0]], basis.clone()),
        Orbital::new(array![[1.0]], basis.clone())
    ];

    let wave_function = SpinDeterminantProduct::new(
        orbitals,
        1
    );

    let kinetic = KineticEnergy::new();
    let potential_ions = IonicPotential::new(ion_pos, array![2]);
    let potential_elec = ElectronicPotential::new();
    let hamiltonian = ElectronicHamiltonian::new(kinetic, potential_ions, potential_elec);

    let rng = StdRng::from_seed([0u8; 32]);
    let metrop = MetropolisDiffuse::from_rng(0.1, rng);

    let mut sampler = Sampler::new(wave_function, metrop);
    sampler.add_observable("Energy", hamiltonian);

    let mut runner = Runner::new(sampler);
    runner.run(1000, 100);

    let energy = *runner.means().get("Energy").unwrap();
    let energy_err = *runner.errors().get("Energy").unwrap();

    let exact_result = 0.5*(1.5_f64).powi(6)*(-0.5);
    assert!((energy - exact_result).abs() < 2.0*energy_err);
}