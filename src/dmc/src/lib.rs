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
        equilibration: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut metrop = MetropolisDiffuse::from_rng(time_step, self.rng.clone());
        let mut energies = vec![self.reference_energy];
        let mut vars = vec![0.0];

        for j in 0..num_iterations {
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
                        .act_on(&self.guiding_wave_function, &new_conf)
                        .unwrap()
                        .get_scalar()
                        .unwrap()
                        / self.guiding_wave_function.value(&new_conf).unwrap();

                    *weight *= f64::exp(
                        -time_step * (0.5 * (local_e + local_e_new) - self.reference_energy),
                    );
                    *conf = new_conf;
                    //ensemble_energy += *weight*local_e_new
                }
                let local_e_new = self
                    .hamiltonian
                    .act_on(&self.guiding_wave_function, conf)
                    .unwrap()
                    .get_scalar()
                    .unwrap()
                    / self.guiding_wave_function.value(conf).unwrap();
                ensemble_energy += *weight * local_e_new
            }

            let global_weight = self.walkers.iter().fold(0.0, |acc, (w, _)| acc + w);
            ensemble_energy /= global_weight;

            // update ref energy
            self.reference_energy = (ensemble_energy + self.reference_energy) / 2.0;

            // perform stochastic reconfiguration
            let new_weight = global_weight / self.walkers.len() as f64;
            let mut confs_weighted: Vec<_> = self
                .walkers
                .iter()
                .map(|(w, c)| Weighted {
                    weight: (w * 100.0) as u32,
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

            // update DMC energy
            if j > equilibration {
                let dmc_energy_prev = *energies.last().unwrap();
                energies.push(
                    dmc_energy_prev
                        + (self.reference_energy - dmc_energy_prev)
                            / (j - equilibration + 2) as f64,
                );
                vars.push(
                    vars.last().unwrap()
                        + ((self.reference_energy - dmc_energy_prev)
                            * (self.reference_energy - energies.last().unwrap())
                            - *vars.last().unwrap())
                            / (j - equilibration + 1) as f64,
                );
                println!(
                    "Reference Energy:   {:.8}    DMC Energy:   {:.8} +/- {:.8}",
                    self.reference_energy,
                    *energies.last().unwrap(),
                    vars.last().unwrap().sqrt() / ((j - equilibration) as f64).sqrt()
                );
            }
        }
        (energies, vars)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
