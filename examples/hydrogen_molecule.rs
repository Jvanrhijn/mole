#[macro_use]
extern crate ndarray;
use rand::{SeedableRng, StdRng};

use basis::GaussianBasis;
use metropolis::MetropolisBox;
use montecarlo::{Runner, Sampler};
use operator::{ElectronicHamiltonian, ElectronicPotential, IonicPotential, KineticEnergy, IonicHamiltonian};
use wavefunction::{JastrowSlater, Orbital};

fn main() {
    let ion_positions = array![[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];

    use basis::Hydrogen1sBasis;
    let basis_set = Hydrogen1sBasis::new(ion_positions.clone(), vec![1.0]);

    let orbitals = vec![
        Orbital::new(array![[1.0], [1.0]], basis_set.clone()),
        Orbital::new(array![[1.0], [1.0]], basis_set.clone())
    ];

    let wave_func = JastrowSlater::new(
        array![5.0],  // parameters
        orbitals,
        0.001, // scale distance
        1 // number of up electrons
    );

    let kinetic = KineticEnergy::new();
    let potential_ions = IonicPotential::new(ion_positions, array![1, 1]);
    let potential_electrons = ElectronicPotential::new();

    let hamiltonian = ElectronicHamiltonian::new(
        kinetic.clone(),
        potential_ions.clone(),
        potential_electrons.clone(),
    );

    let metrop = MetropolisBox::from_rng(1.0, StdRng::from_seed([0; 32]));

    let mut sampler = Sampler::new(wave_func, metrop);
    sampler.add_observable("Energy", hamiltonian);
    sampler.add_observable("Electron potential", potential_electrons);
    sampler.add_observable("Kinetic", kinetic);

    let mut runner = Runner::new(sampler);
    runner.run(1_000_000, 200);

    let total_energy = *runner.means().get("Energy").unwrap();
    let error_energy = *runner.errors().get("Energy").unwrap();

    println!(
        "\nTotal Energy: {e:.*} +/- {s:.*}",
        8,
        8,
        e = total_energy,
        s = error_energy
    );
}
