#[macro_use]
extern crate ndarray;
use basis::GaussianBasis;
use metropolis::MetropolisBox;
use montecarlo::{Runner, Sampler};
use operator::{ElectronicHamiltonian, ElectronicPotential, IonicPotential, KineticEnergy};
use rand::{SeedableRng, StdRng};
use wavefunction::{JastrowSlater, Orbital};

fn main() {
    // setup basis set
    let basis_set = GaussianBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0]);

    // construct orbitals
    let orbital1 = Orbital::new(array![[1.0, 0.5, 0.25]], basis_set.clone());
    let orbital2 = Orbital::new(array![[1.0, 0.5, 0.25]], basis_set);

    // construct Jastrow-Slater wave function
    let wave_function = JastrowSlater::new(
        array![1.0],              // Jastrow factor parameters
        vec![orbital1, orbital2], // orbitals
        0.1,                      // scale distance
        1,                        // number of electrons
    );

    // setup metropolis algorithm/Markov chain generator
    let metrop = MetropolisBox::from_rng(1.0, StdRng::from_seed([0; 32]));

    // Construct kinetic energy and ionic potential operators
    let kinetic = KineticEnergy::new();
    // One ion located at r = (0, 0, 0) with Z = 1
    let potential = IonicPotential::new(array![[0.0, 0.0, 0.0]], array![2]);
    // electron-electron interaction potential
    let potential_ee = ElectronicPotential::new();
    //  Full hamiltonian
    let hamiltonian =
        ElectronicHamiltonian::new(kinetic.clone(), potential.clone(), potential_ee.clone());

    // construct sampler
    let mut sampler = Sampler::new(wave_function, metrop);
    sampler.add_observable("Hamiltonian", hamiltonian);

    // create MC runner
    let mut runner = Runner::new(sampler);

    // Run Monte Carlo integration for 100000 steps, with block size 200
    runner.run(100_000, 500);

    // Retrieve mean values of energy over run
    let energy = *runner.means().get("Hamiltonian").unwrap();
    let var_energy = *runner.variances().get("Hamiltonian").unwrap();

    println!(
        "\nEnergy:         {:.*} +/- {:.*}",
        8,
        energy,
        8,
        var_energy.sqrt()
    );
    println!("Exact ground state energy: -2.903")
}
