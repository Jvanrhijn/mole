// Third party imports
use ndarray::{Array, Array1, Array2, Axis, Ix2};
use ndarray_linalg::Norm;
// First party imports
use crate::traits::{
    Operator,
    OperatorValue::{self, *},
};
use wavefunction::{Cache, Differentiate, Error, Function};

/// Ionic potential energy operator:
/// $\hat{V}_{\mathrm{ion}} = -\sum_{i=1}^{N_{\mathrm{ions}}\sum_{j=1}^{\mathrm{e}} \frac{Z_i}{r_{ij}}$.
/// Ionic charges are in units of the proton charge.
#[derive(Clone)]
pub struct IonicPotential {
    ion_positions: Array2<f64>,
    ion_charge: Array1<i32>,
    ionic_repulsion: f64,
}

impl Function<f64> for IonicPotential {
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        let num_ions = self.ion_positions.len_of(Axis(0));
        let num_elec = cfg.len_of(Axis(0));
        let mut pot = 0.;
        for i in 0..num_ions {
            for j in 0..num_elec {
                let separation = &cfg.slice(s![j, ..]) - &self.ion_positions.slice(s![i, ..]);
                pot -= f64::from(self.ion_charge[i]) / separation.norm_l2();
            }
        }
        Ok(pot + self.ionic_repulsion)
    }
}

impl IonicPotential {
    pub fn new(ion_positions: Array2<f64>, ion_charge: Array1<i32>) -> Self {
        // proton-proton repulsion energy
        let num_ions = ion_positions.shape()[0];
        let mut pot = 0.0;
        for i in 0..num_ions {
            for j in i + 1..num_ions {
                let separation = &ion_positions.slice(s![j, ..]) - &ion_positions.slice(s![i, ..]);
                pot += f64::from(ion_charge[i] * ion_charge[j]) / separation.norm_l2();
            }
        }
        IonicPotential {
            ion_positions,
            ion_charge,
            ionic_repulsion: pot,
        }
    }
}

impl<T: Cache> Operator<T> for IonicPotential {
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue, Error> {
        Ok(Scalar(self.value(cfg)? * wf.current_value().0))
    }
}

/// Electron-electron interaction potential:
/// $\hat{V}_{ee} = \sum_{i=1}^{N_e} \sum_{j>i}^{N_e} \frac{1}{r_{ij}}.$
/// This potential is only a function of the current electronic configuration, and
/// so is not parametrized over anything.
#[derive(Clone, Default)]
pub struct ElectronicPotential;

impl ElectronicPotential {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Function<f64> for ElectronicPotential {
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        let num_elec = cfg.len_of(Axis(0));
        let mut pot = 0.;
        for i in 0..num_elec {
            for j in i + 1..num_elec {
                let separation = &cfg.slice(s![i, ..]) - &cfg.slice(s![j, ..]);
                pot += 1. / separation.norm_l2();
            }
        }
        Ok(pot)
    }
}

impl<T: Cache> Operator<T> for ElectronicPotential {
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue, Error> {
        Ok(Scalar(self.value(cfg)? * wf.current_value().0))
    }
}

/// Kinetic energy operator:
/// $\hat{T} = -\frac{1}{2}\sum_{i=1}^{N_e}\nabla_{i}^2$.
/// The kinetic energy is solely a function of the current electronic configuration,
/// and so it is not parametrized over anything.
#[derive(Clone)]
pub struct KineticEnergy {}

impl Default for KineticEnergy {
    fn default() -> Self {
        KineticEnergy {}
    }
}

impl KineticEnergy {
    pub fn new() -> Self {
        KineticEnergy {}
    }
}

impl<T> Operator<T> for KineticEnergy
where
    T: Differentiate<D = Ix2> + Cache,
{
    fn act_on(&self, wf: &T, _cfg: &Array<f64, Ix2>) -> Result<OperatorValue, Error> {
        Ok(Scalar(-0.5 * wf.current_value().2))
    }
}

/// Ionic Hamiltonian operator:
/// $\hat{H} = \hat{T} + \hat{V}_{\mathrm{ion}}$.
/// Use this for calculations neglecting electron-electron interactions.
#[derive(Clone)]
pub struct IonicHamiltonian {
    v: IonicPotential,
    t: KineticEnergy,
}

impl IonicHamiltonian {
    pub fn new(t: KineticEnergy, v: IonicPotential) -> Self {
        IonicHamiltonian { v, t }
    }
}

impl<T> Operator<T> for IonicHamiltonian
where
    T: Differentiate<D = Ix2> + Cache,
{
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue, Error> {
        Ok(self.t.act_on(wf, cfg)? + self.v.act_on(wf, cfg)?)
    }
}

/// Electronic Hamiltonian operator:
/// $\hat{H} = \hat{T} + \hat{V}_{\mathrm{ion}} + \hat{V}_{\mathrm{ee}}$.
/// Use this operator for constructing a proper local energy operator. The hamiltonian
/// simply acts on any type implementing the WaveFunction and Function traits,
/// but does not care about normalization.
#[derive(Clone)]
pub struct ElectronicHamiltonian {
    t: KineticEnergy,
    vion: IonicPotential,
    velec: ElectronicPotential,
}

impl ElectronicHamiltonian {
    pub fn new(t: KineticEnergy, vion: IonicPotential, velec: ElectronicPotential) -> Self {
        ElectronicHamiltonian { t, vion, velec }
    }
}

impl<T> Operator<T> for ElectronicHamiltonian
where
    T: Differentiate<D = Ix2> + Cache,
{
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue, Error> {
        Ok(self.t.act_on(wf, cfg)? + self.vion.act_on(wf, cfg)? + self.velec.act_on(wf, cfg)?)
    }
}

/// Local energy operator:
/// $\hat{E}_{L}\psi(x) = \frac{\hat{H}\psi}{\psi(x)}$.
/// This operator should be used in any Monte Carlo simulation
/// trying to calculate the electronic ground state of a molecular system.
#[derive(Clone)]
pub struct LocalEnergy<H> {
    h: H,
}

impl<H> LocalEnergy<H> {
    pub fn new(h: H) -> Self {
        LocalEnergy { h }
    }
}

impl<T, H: Operator<T>> Operator<T> for LocalEnergy<H>
where
    T: Differentiate<D = Ix2> + Cache,
{
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<OperatorValue, Error> {
        Ok(&self.h.act_on(wf, cfg)? / &Scalar(wf.current_value().0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use basis::{self, Hydrogen1sBasis, Hydrogen2sBasis};
    use wavefunction::{Orbital, SingleDeterminant};

    #[test]
    fn hydrogen_ground_state() {
        let kinetic = KineticEnergy::new();
        let potential = IonicPotential::new(array![[0., 0., 0.]], array![1]);
        let hamiltonian = IonicHamiltonian::new(kinetic, potential);
        let basis_set = Hydrogen1sBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0]);
        let mut wf = SingleDeterminant::new(vec![Orbital::new(array![[1.0]], basis_set)]);
        let cfg = Array2::<f64>::ones((1, 3));
        wf.refresh(&cfg);
        let hpsi = hamiltonian.act_on(&wf, &cfg).unwrap();
        let wval = wf.value(&cfg).unwrap();
        assert_eq!(&hpsi / &Scalar(wval), Scalar(-0.5));
    }

    #[test]
    fn hydrogen_first_excited() {
        let kinetic = KineticEnergy::new();
        let potential = IonicPotential::new(array![[0., 0., 0.]], array![1]);
        let hamiltonian = IonicHamiltonian::new(kinetic, potential);
        let basis_set = Hydrogen2sBasis::new(array![[0.0, 0.0, 0.0]], vec![2.0]);
        let mut wf = SingleDeterminant::new(vec![Orbital::new(array![[1.0]], basis_set)]);
        let cfg = Array2::<f64>::ones((1, 3));
        wf.refresh(&cfg);
        let hpsi = hamiltonian.act_on(&wf, &cfg).unwrap();
        let wval = wf.value(&cfg).unwrap();
        let hpsi = if let OperatorValue::Scalar(value) = hpsi {
            value
        } else {
            assert!(false);
            0.0
        };
        assert!((hpsi / wval - (-0.125)).abs() < 1e-15);
    }

}
