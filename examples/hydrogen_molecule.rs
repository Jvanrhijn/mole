use std::collections::HashMap;
#[macro_use]
extern crate ndarray;
use rand::{SeedableRng, StdRng};

use metropolis::MetropolisDiffuse;
use montecarlo::{traits::Log, Runner, Sampler};
use operator::{
    ElectronicHamiltonian, ElectronicPotential, IonicPotential, KineticEnergy, OperatorValue,
};
use wavefunction::{JastrowSlater, Orbital};

struct Logger {
    block_size: usize,
}

impl Log for Logger {
    fn log(&self, data: &HashMap<String, Vec<OperatorValue>>) -> String {

        let energy_blocks = &data["Energy"].chunks(self.block_size);

        let block_means = energy_blocks.clone().into_iter().map(|block| {
            block
                .clone()
                .into_iter()
                .fold(OperatorValue::Scalar(0.0), |a, b| a + b.clone())
                / OperatorValue::Scalar(block.len() as f64)
        });

        let energy = block_means.clone().into_iter().sum::<OperatorValue>()
            / OperatorValue::Scalar(block_means.len() as f64);

        format!(
            "Energy: {:.5}    {:.5}",
            energy.get_scalar(),
            block_means.last().unwrap().get_scalar()
        )
    }
}

fn main() {
    let ion_positions = array![[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];

    use basis::Hydrogen1sBasis;
    let basis_set = Hydrogen1sBasis::new(ion_positions.clone(), vec![1.0]);

    let orbitals = vec![
        Orbital::new(array![[1.0], [1.0]], basis_set.clone()),
        Orbital::new(array![[1.0], [1.0]], basis_set.clone()),
    ];

    let wave_func = JastrowSlater::new(
        array![5.0], // parameters
        orbitals,
        0.001, // scale distance
        1,     // number of up electrons
    );

    let kinetic = KineticEnergy::new();
    let potential_ions = IonicPotential::new(ion_positions, array![1, 1]);
    let potential_electrons = ElectronicPotential::new();

    let hamiltonian = ElectronicHamiltonian::new(
        kinetic.clone(),
        potential_ions.clone(),
        potential_electrons.clone(),
    );

    let metrop = MetropolisDiffuse::from_rng(0.1, StdRng::from_seed([0; 32]));

    let mut sampler = Sampler::new(wave_func, metrop);
    sampler.add_observable("Energy", hamiltonian);
    //sampler.add_observable("Electron potential", potential_electrons);
    //sampler.add_observable("Kinetic", kinetic);

    let steps = 1_000_000;
    let block_size = 100;

    let mut runner = Runner::new(sampler, Logger { block_size });
    runner.run(steps, block_size);

    //let total_energy = *runner.means().get("Energy").unwrap();
    //let error_energy = *runner.errors().get("Energy").unwrap();

    //println!(
    //    "\nTotal Energy: {e:.*} +/- {s:.*}",
    //    8,
    //    8,
    //    e = total_energy,
    //    s = error_energy
    //);
}
