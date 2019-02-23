#[macro_use]
extern crate ndarray;
use basis::{GaussianBasis};
use metropolis::MetropolisBox;
use montecarlo::{Runner, Sampler};

use operator::{ElectronicHamiltonian, ElectronicPotential, IonicPotential, KineticEnergy};
use rand::{SeedableRng, StdRng};
use wavefunction::{Orbital, SingleDeterminant};

fn main() {
    let ion_positions = array![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let basis_set = GaussianBasis::new(ion_positions.clone(), vec![1.0, 2.0, 3.0]);

    let orbital1 = Orbital::new(array![[1.0, 0.5, 0.25], [0.0, 0.0, 0.0]], basis_set.clone());
    let orbital2 = Orbital::new(array![[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]], basis_set);

    let wave_func = SingleDeterminant::new(vec![orbital1, orbital2]);

    let kinetic = KineticEnergy::new();
    let potential_ions = IonicPotential::new(ion_positions, array![1, 1]);
    let potential_electrons = ElectronicPotential::new();
    let hamiltonian = ElectronicHamiltonian::new(
        kinetic.clone(),
        potential_ions.clone(),
        potential_electrons.clone(),
    );

    let metrop = MetropolisBox::from_rng(0.15, StdRng::from_seed([0; 32]));

    let mut sampler = Sampler::new(wave_func, metrop);
    sampler.add_observable("Kinetic Energy", kinetic);
    sampler.add_observable("Ion Potential", potential_ions);
    sampler.add_observable("Electron Potential", potential_electrons);
    sampler.add_observable("Energy", hamiltonian);

    let mut runner = Runner::new(sampler);
    runner.run(100_000, 100);

    let total_energy = *runner.means().get("Energy").unwrap();
    let energy_variance = *runner.variances().get("Energy").unwrap();

    println!(
        "\nTotal Energy: {e:.*} +/- {s:.*}",
        8,
        8,
        e = total_energy,
        s = energy_variance.sqrt()
    );
}
