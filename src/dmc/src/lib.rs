use metropolis::{Metropolis, MetropolisDiffuse};
use ndarray::{Array2, Ix2};
use ndarray_rand::RandomExt;
use operator::LocalOperator;
use rand::distributions::Normal;
use rand::distributions::{Distribution, Weighted, WeightedChoice};
use rand::{FromEntropy, RngCore, SeedableRng, StdRng};
use wavefunction_traits::*;

// TODO: custom logging, parallelization, allow different
// branch-split algorithms, allow release-node scheme, allow
// arbitrary sampling of observables

pub struct DmcRunner<T, O, R>
where
    T: Function<f64, D = Ix2> + Differentiate + WaveFunction + Clone,
    O: LocalOperator<T>,
    R: SeedableRng,
{
    guiding_wave_function: T,
    walkers: Vec<(f64, Array2<f64>)>,
    reference_energy: f64,
    hamiltonian: O,
    rng: R,
}

impl<T, O, R> DmcRunner<T, O, R>
where
    T: Function<f64, D = Ix2> + Differentiate<D = Ix2> + WaveFunction + Clone,
    O: LocalOperator<T>,
    R: SeedableRng + RngCore + Clone,
    <R as SeedableRng>::Seed: From<[u8; 32]>,
{
    pub fn with_rng(
        guiding_wave_function: T,
        num_walkers: usize,
        reference_energy: f64,
        hamiltonian: O,
        mut rng: R,
    ) -> Self {
        let mut confs = vec![];
        for _ in 0..num_walkers {
            confs.push((
                1.0,
                Array2::random_using(
                    (guiding_wave_function.num_electrons(), 3),
                    Normal::new(0.0, 1.0),
                    &mut rng,
                ),
            ));
        }
        Self {
            guiding_wave_function,
            walkers: confs,
            reference_energy,
            hamiltonian,
            rng,
        }
    }

    pub fn diffuse(
        &mut self,
        time_step: f64,
        num_iterations: usize,
        block_size: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut metrop = MetropolisDiffuse::from_rng(time_step, self.rng.clone());
        let mut energies = vec![];
        let mut vars = vec![];

        let blocks = num_iterations / block_size;

        for block_nr in 0..blocks {
            let mut energies_block = vec![];
            for j in 0..block_size {
                let mut ensemble_energy = 0.0;

                for (weight, conf) in self.walkers.iter_mut() {
                    for e in 0..self.guiding_wave_function.num_electrons() {
                        let mut new_conf = if let Some(x) = metrop
                            .move_state(&mut self.guiding_wave_function, conf, e)
                            .unwrap()
                        {
                            x
                        } else {
                            conf.clone()
                        };

                        // apply FN approximation
                        if self.guiding_wave_function.value(conf).unwrap().signum()
                            != self
                                .guiding_wave_function
                                .value(&new_conf)
                                .unwrap()
                                .signum()
                        {
                            new_conf = conf.clone();
                        }
                        *conf = new_conf;
                    }
                    // compute weight
                    let local_e = self
                        .hamiltonian
                        .act_on(&self.guiding_wave_function, &conf)
                        .unwrap()
                        .get_scalar()
                        .unwrap()
                        / self.guiding_wave_function.value(&conf).unwrap();
                    let local_e_new = self
                        .hamiltonian
                        .act_on(&self.guiding_wave_function, conf)
                        .unwrap()
                        .get_scalar()
                        .unwrap()
                        / self.guiding_wave_function.value(conf).unwrap();
                    *weight *= f64::exp(
                        -time_step * (0.5 * (local_e + local_e_new) - self.reference_energy),
                    );
                    ensemble_energy += *weight * local_e_new
                }

                let global_weight = self.walkers.iter().fold(0.0, |acc, (w, _)| acc + w);
                ensemble_energy /= global_weight;

                // update ref energy
                //self.reference_energy = (ensemble_energy + self.reference_energy) / 2.0;
                energies_block.push(ensemble_energy);

                // perform stochastic reconfiguration
                let new_weight = global_weight / self.walkers.len() as f64;
                let max_weight = self.walkers.iter().fold(0.0, |acc, (w, _)| f64::max(acc, *w));
                // normalize weights by the maximum weight
                let norm_factor = self.walkers.len() as f64 / max_weight;
                let mut confs_weighted: Vec<_> = self
                    .walkers
                    .iter()
                    .map(|(w, c)| Weighted {
                        weight: (w*norm_factor) as u32,
                        item: c,
                    })
                    .collect();
                let wc = WeightedChoice::new(&mut confs_weighted);
                // construct new walkers
                let mut new_walkers = vec![];
                for _ in 0..self.walkers.len() {
                    new_walkers.push((new_weight, wc.sample(&mut self.rng).clone()));
                }
                self.walkers = new_walkers;
            }
            if block_nr == 1 {
                let energy = energies_block.iter().sum::<f64>() / energies_block.len() as f64;
                //self.reference_energy = (self.reference_energy + energy) / 2.0;
                self.reference_energy = energy;
                energies.push(energy);
                vars.push(0.0);
                println!(
                        "Block Energy:   {:.8}    DMC Energy:   {:.8} +/- {:.8}",
                        energy,
                        *energies.last().unwrap(),
                        0.0
                    );
            }
            if block_nr > 1 {
                let energy = energies_block.iter().sum::<f64>() / energies_block.len() as f64;
                let dmc_energy_prev = *energies.last().unwrap();
                energies.push(
                    dmc_energy_prev
                        + (energy - dmc_energy_prev)
                            / (block_nr + 1) as f64,
                );
                vars.push(
                    vars.last().unwrap()
                        + ((energy - dmc_energy_prev)
                            * (energy - energies.last().unwrap())
                            - *vars.last().unwrap())
                            / (block_nr + 1) as f64,
                );
                println!(
                        "Block Energy:   {:.8}    DMC Energy:   {:.8} +/- {:.8}",
                        energy,
                        *energies.last().unwrap(),
                        vars.last().unwrap().sqrt()/(block_nr as f64).sqrt(),
                    );
            }
        }
        (energies, vars.iter().enumerate().map(|(i, x)| (x / ((i + 1) as f64)).sqrt()).collect())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
