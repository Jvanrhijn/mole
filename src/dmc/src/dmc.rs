use metropolis::{Metropolis, MetropolisDiffuse};
use ndarray::{Array2, Ix2};
use ndarray_rand::RandomExt;
use operator::LocalOperator;
use rand::distributions::Normal;
use rand::distributions::{Distribution, Weighted, WeightedChoice};
use rand::{FromEntropy, RngCore, SeedableRng, StdRng};
use wavefunction_traits::*;

use crate::traits::BranchingAlgorithm;

// TODO: custom logging, parallelization, allow different
// branch-split algorithms, allow release-node scheme, allow
// arbitrary sampling of observables

pub struct DmcRunner<T, O, R, B>
where
    T: Function<f64, D = Ix2> + Differentiate<D = Ix2> + WaveFunction + Clone,
    O: LocalOperator<T>,
    R: SeedableRng + RngCore + Clone,
    B: BranchingAlgorithm<R>,
    <R as SeedableRng>::Seed: From<[u8; 32]>,
{
    guiding_wave_function: T,
    walkers: Vec<(f64, Array2<f64>)>,
    reference_energy: f64,
    hamiltonian: O,
    metrop: MetropolisDiffuse<R>,
    branching: B,
}

impl<T, O, R, B> DmcRunner<T, O, R, B>
where
    T: Function<f64, D = Ix2> + Differentiate<D = Ix2> + WaveFunction + Clone,
    O: LocalOperator<T>,
    R: SeedableRng + RngCore + Clone,
    B: BranchingAlgorithm<R>,
    <R as SeedableRng>::Seed: From<[u8; 32]>,
{
    pub fn new(
        guiding_wave_function: T,
        num_walkers: usize,
        reference_energy: f64,
        hamiltonian: O,
        mut metropolis: MetropolisDiffuse<R>,
        branching: B,
    ) -> Self {
        //let mut confs = vec![];
        let mut rng = <MetropolisDiffuse<R> as Metropolis<T>>::rng_mut(&mut metropolis);
        let confs = vec![
            (
                1.0, 
                Array2::random_using(
                    (guiding_wave_function.num_electrons(), 3),
                    Normal::new(0.0, 1.0),
                    &mut rng,
                )
            ); num_walkers
        ];
        Self {
            guiding_wave_function,
            walkers: confs,
            reference_energy,
            hamiltonian,
            metrop: metropolis,
            branching,
        }
    }

    pub fn diffuse(
        &mut self,
        time_step: f64,
        num_iterations: usize,
        block_size: usize,
        num_eq_blocks: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut energies = vec![];
        let mut vars = vec![];

        let blocks = num_iterations / block_size;

        for block_nr in 0..blocks {
            let mut energies_block = vec![];
            for j in 0..block_size {
                let mut total_weight = 0.0;
                let mut ensemble_energy = 0.0;

                for (weight, conf) in self.walkers.iter_mut() {
                    // compute old local energy
                    let wave_function_value_old = self.guiding_wave_function.value(&conf).unwrap();
                    let local_e = self
                        .hamiltonian
                        .act_on(&self.guiding_wave_function, &conf)
                        .unwrap()
                        .get_scalar()
                        .unwrap()
                        / wave_function_value_old;
                    // move all electrons according to Langevin dynamics
                    // with accept/reject
                    for e in 0..self.guiding_wave_function.num_electrons() {
                        let new_conf = if let Some(x) = self
                            .metrop
                            .move_state(&mut self.guiding_wave_function, conf, e)
                            .unwrap()
                        {
                            x
                        } else {
                            conf.clone()
                        };
                        *conf = new_conf;
                    }

                    ensemble_energy += *weight * local_e;
                    total_weight += *weight;

                    let wave_function_value_new = self.guiding_wave_function.value(&conf).unwrap();

                    // compute local energy after move
                    let local_e_new = self
                        .hamiltonian
                        .act_on(&self.guiding_wave_function, conf)
                        .unwrap()
                        .get_scalar()
                        .unwrap()
                        / wave_function_value_new;
                    // update weight of this walker
                    *weight *= f64::exp(
                        -time_step * ((local_e + local_e_new)/2.0 - self.reference_energy),
                    );
                    //ensemble_energy += *weight * local_e_new;
                }

                //let total_weight = self.walkers.iter().fold(0.0, |acc, (w, _)| acc + w);
                ensemble_energy /= total_weight;

                energies_block.push(ensemble_energy);

                // perform branching step
                //self.walkers = new_walkers;
                let mut rng = <MetropolisDiffuse<R> as Metropolis<T>>::rng_mut(&mut self.metrop);
                self.walkers = self.branching.branch(&self.walkers, &mut rng);
            }
            // update reference energy and store
            // TODO: cleaner solution
            self.update_energies(&energies_block, &mut energies, &mut vars, block_nr, num_eq_blocks);
        }
        (
            energies,
            vars.iter()
                .enumerate()
                .map(|(i, x)| (x / ((i + 1) as f64)).sqrt())
                .collect(),
        )
    }

    fn update_energies(
        &mut self,
        energies_block: &Vec<f64>,
        energies: &mut Vec<f64>,
        vars: &mut Vec<f64>,
        block_nr: usize,
        num_eq_blocks: usize,
    ) {
        if block_nr == num_eq_blocks {
            let energy = energies_block.iter().sum::<f64>() / energies_block.len() as f64;
            // mix reference and current energy for better convergence
            self.reference_energy = (self.reference_energy + energy) / 2.0;
            energies.push(energy);
            // initialize variance to zero
            vars.push(0.0);
            // pretty printing
            println!(
                "Block Energy:   {:.8}    DMC Energy:   {:.8} +/- {:.8}",
                energy,
                *energies.last().unwrap(),
                0.0
            );
        }
        if block_nr > num_eq_blocks {
            // average energy over the last block
            let energy = energies_block.iter().sum::<f64>() / energies_block.len() as f64;
            let dmc_energy_prev = *energies.last().unwrap();
            // running mean formula
            energies.push(dmc_energy_prev + (energy - dmc_energy_prev) / (block_nr - num_eq_blocks) as f64);
            // mix reference and current energy for better convergence
            self.reference_energy = (self.reference_energy + *energies.last().unwrap()) / 2.0;
            // update variance
            vars.push(
                vars.last().unwrap()
                    + ((energy - dmc_energy_prev) * (energy - energies.last().unwrap())
                        - *vars.last().unwrap())
                        / (block_nr - num_eq_blocks) as f64,
            );
            let err = vars.last().unwrap().sqrt() / ((block_nr - num_eq_blocks) as f64).sqrt();
            // pretty printing
            println!(
                "Block Energy:   {:.8}    DMC Energy:   {:.8} +/- {:.8}",
                energy,
                *energies.last().unwrap(),
                err,
            );
        }
    }
}
