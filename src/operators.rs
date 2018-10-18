// Third party imports
use ndarray::{Array, Array1, Array2, Ix2, Axis};
// First party imports
use traits::operator::Operator;
use traits::function::Function;
use traits::wavefunction::WaveFunction;
use error::{Error};

// Ionic potential energy operator

pub struct IonicPotential {
    ion_positions: Array2<f64>,
    ion_charge: Array1<i32>
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
                let distance = separation.dot(&separation).sqrt();
                pot -= self.ion_charge[i] as f64/distance;
            }
        }
        Ok(pot)
    }

}

impl IonicPotential {
    pub fn new(ion_positions: Array2<f64>, ion_charge: Array1<i32>) -> Self {
        IonicPotential{ion_positions, ion_charge}
    }
}

impl<T> Operator<T> for IonicPotential
where T: Function<f64, D=Ix2> + ?Sized,
{

    type V = f64;

    fn act_on(&self, wf: &T, cfg: &Array2<Self::V>) -> Result<Self::V, Error> {
        Ok(self.value(cfg)?*wf.value(cfg)?)
    }
}

pub struct ElectronicPotential {}

#[allow(dead_code)]
impl ElectronicPotential {
    pub fn new() -> Self {
        ElectronicPotential{}
    }
}

impl Function<f64> for ElectronicPotential {

    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        let num_elec = cfg.len_of(Axis(0));
        let mut pot = 0.;
        for i in 0..num_elec {
            for j in i+1..num_elec {
                let separation = &cfg.slice(s![i, ..]) - &cfg.slice(s![j, ..]);
                pot += 1./separation.dot(&separation).sqrt();
            }
        }
        Ok(pot)
    }
}

impl<T> Operator<T> for ElectronicPotential
where T: Function<f64, D=Ix2> + ?Sized,
{
    type V = f64;

    fn act_on(&self, wf: &T, cfg: &Array2<Self::V>) -> Result<Self::V, Error> {
        Ok(self.value(cfg)?*wf.value(cfg)?)
    }
}

// Kinetic energy operator

pub struct KineticEnergy {}

impl KineticEnergy {
    pub fn  new() -> Self {
        KineticEnergy{}
    }
}

impl<T> Operator<T> for KineticEnergy
where T: Function<f64, D=Ix2> + WaveFunction<D=Ix2> + ?Sized,
{
    type V = f64;

    fn act_on(&self, wf: &T, cfg: &Array<Self::V, Ix2>) -> Result<Self::V, Error> {
        Ok(-0.5*wf.laplacian(cfg)?)
    }
}

// Ionic Hamiltonian operator

pub struct IonicHamiltonian {
    v: IonicPotential,
    t: KineticEnergy
}

#[allow(dead_code)]
impl IonicHamiltonian {
    pub fn new(t: KineticEnergy, v: IonicPotential) -> Self {
        IonicHamiltonian{v, t}
    }
}

impl<T> Operator<T> for IonicHamiltonian
where T: Function<f64, D=Ix2> + WaveFunction<D=Ix2> + ?Sized,
{
    type V = f64;

    fn act_on(&self, wf: &T, cfg: &Array2<Self::V>) -> Result<Self::V, Error> {
        Ok(self.t.act_on(wf, cfg)? + self.v.act_on(wf, cfg)?)
    }
}

pub struct ElectronicHamiltonian {
    t: KineticEnergy,
    vion: IonicPotential,
    velec: ElectronicPotential
}

#[allow(dead_code)]
impl ElectronicHamiltonian {
    pub fn new(t: KineticEnergy, vion: IonicPotential, velec: ElectronicPotential) -> Self {
        ElectronicHamiltonian{t, vion, velec}
    }
}

impl<T> Operator<T> for ElectronicHamiltonian
where T: Function<f64, D=Ix2> + WaveFunction<D=Ix2> + ?Sized
{
    type V = f64;

    fn act_on(&self, wf: &T, cfg: &Array2<Self::V>) -> Result<Self::V, Error> {
        Ok(self.t.act_on(wf, cfg)? + self.vion.act_on(wf, cfg)? + self.velec.act_on(wf, cfg)?)
    }
}

// Local energy
pub struct LocalEnergy {
    h: ElectronicHamiltonian
}

impl LocalEnergy {
    pub fn new(h: ElectronicHamiltonian) -> Self {
        LocalEnergy{h}
    }
}

impl<T> Operator<T> for LocalEnergy
where T: Function<f64, D=Ix2> + WaveFunction<D=Ix2> + ?Sized
{
    type V = f64;

    fn act_on(&self, wf: &T, cfg: &Array2<Self::V>) -> Result<Self::V, Error> {
        Ok(self.h.act_on(wf, cfg)?/wf.value(cfg)?)
    }
}
