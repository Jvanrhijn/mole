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
    energy: f64,
}

impl Logger {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            energy: 0.0,
        }
    }

    fn compute_mean_and_block_avg(&self, name: &str, data: &HashMap<String, Vec<OperatorValue>>) -> (f64, f64) {
        let blocks = &data[name].chunks(self.block_size);

        let block_means = blocks.clone().into_iter().map(|block| {
            block
                .clone()
                .into_iter()
                .fold(OperatorValue::Scalar(0.0), |a, b| a + b.clone())
                / OperatorValue::Scalar(block.len() as f64)
        });

        let quantity = *(block_means.clone().into_iter().sum::<OperatorValue>()
            / OperatorValue::Scalar(block_means.len() as f64))
        .get_scalar()
        .unwrap();

        (quantity, *block_means.last().unwrap().get_scalar().unwrap())

    }
}

impl Log for Logger {
    fn log(&mut self, data: &HashMap<String, Vec<OperatorValue>>) -> String {
        let (energy, energy_ba) = self.compute_mean_and_block_avg("Energy", data);
        let (ke, ke_ba) = self.compute_mean_and_block_avg("Kinetic", data);
        let (pe, pe_ba) = self.compute_mean_and_block_avg("Electron potential", data);

        format!(
            "Energy: {:.5}  {:.5}    Kinetic: {:.5}    Electron Potential: {:.5}",
            energy,
            energy_ba,
            ke,
            pe
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

    let metrop = MetropolisDiffuse::from_rng(0.5, StdRng::from_seed([0; 32]));

    let mut sampler = Sampler::new(wave_func, metrop);
    sampler.add_observable("Energy", hamiltonian);
    sampler.add_observable("Electron potential", potential_electrons);
    sampler.add_observable("Kinetic", kinetic);

    let steps = 1_000_000;
    let block_size = 50;

    let mut runner = Runner::new(sampler, Logger::new(block_size));
    runner.run(steps, block_size);
}
