#[macro_use]
extern crate ndarray;
use basis::GaussianBasis;
use metropolis::MetropolisBox;
use montecarlo::{Runner, Sampler};
use operator::{ElectronicPotential, ElectronicHamiltonian, IonicPotential, KineticEnergy};
use rand::{SeedableRng, StdRng};
use wavefunction::{Orbital, SingleDeterminant, JastrowFactor, JastrowSlater};

fn main() {
    // setup basis set
    let basis_set = GaussianBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0, 2.0, 3.0]);

    // construct orbital
    let orbital1 = Orbital::new(array![[1.0, 0.0, 0.0]], basis_set.clone());
    let orbital2 = Orbital::new(array![[1.0, 0.0, 0.0]], basis_set);

    // construct Jastrow-Slater wave function
    let det_up = SingleDeterminant::new(vec![orbital1]);
    let det_down = SingleDeterminant::new(vec![orbital2]);
    let jas = JastrowFactor::new(array![0.5, 0.1], 2);
    let wave_function = JastrowSlater::new(det_up, det_down, jas);

    // setup metropolis algorithm/Markov chain generator
    // We'll use a simple Metropolis-Hastings algorithm with
    // proposal probability described by the Green function of
    // Fokker-Planck equation, with step size 0.5
    let metrop = MetropolisBox::from_rng(1.0, StdRng::from_seed([0; 32]));

    // Construct kinetic energy and ionic potential operators
    let kinetic = KineticEnergy::new();
    // One ion located at r = (0, 0, 0) with Z = 1
    let potential = IonicPotential::new(array![[0.0, 0.0, 0.0]], array![2]);
    // electron-electron interaction potential
    let potential_ee = ElectronicPotential::new();
    //  Full hamiltonian
    let hamiltonian = ElectronicHamiltonian::new(kinetic.clone(), potential.clone(), potential_ee.clone());

    // construct sampler
    let mut sampler = Sampler::new(wave_function, metrop);
    sampler.add_observable("Hamiltonian", hamiltonian);
    sampler.add_observable("Kinetic Energy", kinetic);
    sampler.add_observable("Potential Energy", potential);
    sampler.add_observable("Interaction Potential", potential_ee);

    // create MC runner
    let mut runner = Runner::new(sampler);

    // Run Monte Carlo integration for 100000 steps, with block size 200
    runner.run(100_00, 100);

    // Retrieve mean values of operators over MC run
    let ke = *runner.means().get("Kinetic Energy").unwrap();
    let pe = *runner.means().get("Potential Energy").unwrap();
    let pe_ee = *runner.means().get("Interaction Potential").unwrap();
    let energy = *runner.means().get("Hamiltonian").unwrap();

    let var_ke = *runner.variances().get("Kinetic Energy").unwrap();
    let var_pe = *runner.variances().get("Potential Energy").unwrap();
    let var_pe_ee = *runner.variances().get("Interaction Potential").unwrap();
    let var_energy = *runner.variances().get("Hamiltonian").unwrap();

    println!("");
    println!(
        "Kinetic energy:    {:.*} +/- {:.*}",
        8,
        ke,
        8,
        var_ke.sqrt()
    );
    println!("Potential energy: {:.*} +/- {:.*}", 8, pe, 8, var_pe.sqrt());
    println!("Interaction Energy: {:.*} +/- {:.*}", 8, pe_ee, 8, var_pe_ee.sqrt());
    println!(
        "Energy:     {:.*} +/- {:.*}",
        8,
        energy,
        8,
        var_energy.sqrt()
    );
}
