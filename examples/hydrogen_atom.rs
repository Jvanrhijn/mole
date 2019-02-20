#[macro_use]
extern crate ndarray;
use montecarlo::{Runner, Sampler};
use basis::{gaussian, Func};
use wavefunction::{Orbital, SingleDeterminant};
use metropolis::MetropolisBox;
use operator::{IonicPotential, IonicHamiltonian, KineticEnergy};

fn main() {
    // setup basis set
    let basis_set: Vec<Box<Func>> = vec![
        Box::new(|x| gaussian(x, 1.0)),
        Box::new(|x| gaussian(x, 2.0)),
        Box::new(|x| gaussian(x, 3.0)),
    ];

    // construct orbital
    let orbital = Orbital::new(array![1.0, 0.5, 0.25], &basis_set);

    // construct Slater-determinant wave function
    let wave_function = SingleDeterminant::new(vec![orbital]);

    // setup metropolis algorithm/Markov chain generator
    // We'll use a simple Metropolis-Hastings algorithm with
    // proposal probability which is uniform over a box
    // of size 1 x 1 x 1.
    let metrop = MetropolisBox::new(1.0);

    // Construct kinetic energy and ionic potential operators
    let kinetic = KineticEnergy::new();
    // One ion located at r = (0, 0, 0) with Z = 1
    let potential = IonicPotential::new(array![[0.0, 0.0, 0.0]], array![1]);
    // Ionic Hamiltonian: H = T + V
    //let hamiltonian = IonicHamiltonian::new(kinetic, potential);

    // construct sampler
    let mut sampler = Sampler::new(wave_function, metrop);
    //sampler.add_observable("Hamiltonian", hamiltonian);
    sampler.add_observable("Kinetic Energy", kinetic);
    sampler.add_observable("Potential Energy", potential);

    // create MC runner
    let mut runner = Runner::new(sampler);
    
    // Run Monte Carlo integration for 100000 steps, with block size 10
    runner.run(100_000, 200);

    // Retrieve mean values of operators over MC run
    let ke = *runner.means().get("Kinetic Energy").unwrap();
    let pe = *runner.means().get("Potential Energy").unwrap();
    let energy = ke + pe;

    let var_ke = *runner.variances().get("Kinetic Energy").unwrap();
    let var_pe = *runner.variances().get("Potential Energy").unwrap();
    let var_energy = var_ke + var_pe;
    
    println!("");
    println!("Kinetic energy:   {:.*} +/- {:.*}", 8, ke, 8, var_ke.sqrt());
    println!("Potential energy: {:.*} +/- {:.*}", 8, pe, 8, var_pe.sqrt());
    println!("Total Energy:     {:.*} +/- {:.*}", 8, energy, 8, var_energy.sqrt());

}

