use std::collections::HashMap;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate ndarray;
use basis::Hydrogen1sBasis;
use metropolis::MetropolisDiffuse;
use montecarlo::{traits::Log, Runner, Sampler};
use ndarray::{Array1, Array2, Axis};
use operator::{
    ElectronicHamiltonian, ElectronicPotential, IonicPotential, KineticEnergy, OperatorValue,
    ParameterGradient, WavefunctionValue
};
use rand::{SeedableRng, StdRng};
use wavefunction::{JastrowSlater, Orbital};

struct Logger {
    block_size: usize,
}

impl Logger {
    pub fn new(block_size: usize) -> Self {
        Self { block_size }
    }

    fn compute_mean_and_block_avg(
        &self,
        name: &str,
        data: &HashMap<String, Vec<OperatorValue>>,
    ) -> (f64, f64) {
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
        //let (energy, energy_ba) = self.compute_mean_and_block_avg("Hamiltonian", data);
        //format!("Energy: {:.5}  {:.5}", energy, energy_ba,)
        format!("Parameter gradient: {}", data.get("Parameter gradient").unwrap().last().unwrap())
    }
}

fn main() {
    let optimal_width = 1.0 / 1.69;
    // setup basis set
    let ion_pos = array![[1.0, 0.0, 0.0]];
    let basis_set = Hydrogen1sBasis::new(ion_pos.clone(), vec![optimal_width]);

    // construct orbitals
    let orbitals = vec![
        Orbital::new(array![[1.0]], basis_set.clone()),
        Orbital::new(array![[1.0]], basis_set.clone()),
    ];

    let jas_parm = array![0.7, -0.01, -0.15];
    let jas_parm = array![0.715032069039397, -0.008284096379704453, -0.13916419213978137];
    let jas_parm = array![0.72896599726586, -0.0063307924160986035, -0.11352181747970827];
    let jas_parm = array![0.7400272262239866, -0.004675499869716436, -0.10826331288969135];
    let jas_parm = array![0.7513248278495468, -0.003260195642980996, -0.10697714046006748];
    // construct Jastrow-Slater wave function
    let wave_function = JastrowSlater::new(
        jas_parm.clone(), // Jastrow factor parameters
        orbitals,
        0.001, // scale distance
        1,     // number of electrons with spin up
    );

    // setup metropolis algorithm/markov chain generator
    let metrop = MetropolisDiffuse::from_rng(0.1, StdRng::from_seed([0; 32]));

    // Construct kinetic energy and ionic potential operators
    let kinetic = KineticEnergy::new();
    // One ion located at r = (0, 0, 0) with Z = 1
    let potential = IonicPotential::new(ion_pos, array![2]);
    // electron-electron interaction potential
    let potential_ee = ElectronicPotential::new();
    //  Full hamiltonian
    let hamiltonian =
        ElectronicHamiltonian::new(kinetic.clone(), potential.clone(), potential_ee.clone());

    // construct sampler
    let mut sampler = Sampler::new(wave_function, metrop);
    sampler.add_observable("Hamiltonian", hamiltonian);
    sampler.add_observable("Parameter gradient", ParameterGradient);
    sampler.add_observable("Wavefunction value", WavefunctionValue);

    let block_size = 100;
    let steps = 1_00_00;

    // create MC runner
    let mut runner = Runner::new(sampler, Logger::new(block_size));

    // Run Monte Carlo integration for 100000 steps, with block size 50
    runner.run(steps, block_size);

    let energy_data = Array1::<f64>::from_vec(
        runner
            .data()
            .get("Hamiltonian")
            .unwrap()
            .iter()
            .map(|x| *x.get_scalar().unwrap())
            .collect::<Vec<_>>(),
    );

    // Retrieve mean values of energy over run
    let energy = *energy_data.mean_axis(Axis(0)).first().unwrap();
    let energy_err =
        *energy_data.std_axis(Axis(0), 0.0).first().unwrap() / ((block_size * steps) as f64).sqrt();

    let par_grads = runner.data().get("Parameter gradient").unwrap();
    let local_energy = runner.data().get("Hamiltonian").unwrap();
    let wf_values = runner.data().get("Wavefunction value").unwrap();

    let local_energy_grad = izip!(par_grads, local_energy, wf_values)
        .map(|(psi_i, el, psi)| 2.0 * psi_i.get_vector().unwrap() / *psi.get_scalar().unwrap() * (el.get_scalar().unwrap() - energy))
        .collect::<Vec<Array1<f64>>>();

    let energy_grad = local_energy_grad.iter().fold(Array1::zeros(3), |a, b| a + b) / (local_energy.len() as f64);

    
    println!(
        "\nEnergy:         {:.*} +/- {:.*}",
        8, energy, 8, energy_err
    );

    println!("Exact ground state energy: -2.903");

    println!("\nEnergy gradient: {}", energy_grad);

    let step_size = 1e-4;
    let new_parm = jas_parm + step_size * energy_grad;

    println!("\nSuggested new parameters: {}", new_parm);
}
