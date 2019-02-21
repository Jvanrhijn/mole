#[macro_use]
extern crate ndarray;
use ndarray::{Array1, Array2};
use rand::{StdRng, SeedableRng};
use montecarlo::{Runner, Sampler};
use basis::{gaussian, Func};
use wavefunction::{Orbital, SingleDeterminant};
use metropolis::{MetropolisDiffuse};
use operator::{IonicPotential, KineticEnergy, ElectronicPotential, ElectronicHamiltonian};


fn construct_basis(ion_positions: &'static [[f64; 3]; 2]) -> Vec<Box<Func>> {
    let mut basis_set: Vec<Box<Func>> = Vec::new();
    for position in ion_positions.iter() {
        basis_set.push(Box::new(move |x| gaussian(&(x - &Array1::from_vec(position.to_vec())), 1.0)));
        basis_set.push(Box::new(move |x| gaussian(&(x - &Array1::from_vec(position.to_vec())), 2.0)));
        basis_set.push(Box::new(move |x| gaussian(&(x - &Array1::from_vec(position.to_vec())), 3.0)));
    }
    basis_set
}


fn main() {
    static ION_POSITIONS: [[f64; 3]; 2] = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let basis_set: Vec<Box<Func>> = construct_basis(&ION_POSITIONS);

    let orbital1 = Orbital::new(array![1.0, 0.5, 0.25, 0.0, 0.0, 0.0], &basis_set);
    let orbital2 = Orbital::new(array![0.0, 0.0, 0.0, 1.0, 0.5, 0.25], &basis_set);

    let wave_func = SingleDeterminant::new(vec![orbital1, orbital2]);

    let ion_positions_array = Array2::from_shape_vec((2, 3),
                               ION_POSITIONS.iter().flat_map(|x| x.iter()).cloned().collect()).unwrap();

    let kinetic = KineticEnergy::new();
    let potential_ions = IonicPotential::new(ion_positions_array, array![1, 1]);
    let potential_electrons = ElectronicPotential::new();
    let hamiltonian = ElectronicHamiltonian::new(kinetic.clone(), potential_ions.clone(), potential_electrons.clone());

    let metrop = MetropolisDiffuse::from_rng(0.15, StdRng::from_seed([0; 32]));

    let mut sampler = Sampler::new(wave_func, metrop);
    sampler.add_observable("Kinetic Energy", kinetic);
    sampler.add_observable("Ion Potential", potential_ions);
    sampler.add_observable("Electron Potential", potential_electrons);
    sampler.add_observable("Energy", hamiltonian);

    let mut runner = Runner::new(sampler);
    runner.run(100_000, 100);

    let total_energy = *runner.means().get("Energy").unwrap();
    let energy_variance = *runner.variances().get("Energy").unwrap();

    println!("\nTotal Energy: {e:.*} +/- {s:.*}", 8, 8, e=total_energy, s=energy_variance.sqrt());
}
