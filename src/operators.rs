use ndarray::{Array, Array1, Array2, Ix1, Ix2, Axis};

use traits::operator::Operator;
use traits::function::Function;
use wf::SingleDeterminant;


pub struct IonicPotential {
    ion_positions: Array2<f64>,
    ion_charge: Array1<i32>
}

impl Function<f64> for IonicPotential {
    type E = ();
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Self::E> {
        let num_ions = self.ion_positions.len_of(Axis(0));
        let num_elec = cfg.len_of(Axis(0));
        let mut pot = 0.;
        for i in 0..num_ions {
            for j in 0..num_elec {
                let cur_elec_slice = cfg.slice(s![j, ..]);
                let cur_ion_slice = self.ion_positions.slice(s![i, ..]);
                let separation = &cur_elec_slice - &cur_ion_slice;
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
where T: Function<f64, D=Ix2>
{

    type V = f64;

    fn act_on(&self, wf: &T, cfg: &Array2<Self::V>) -> Self::V {
        self.value(cfg).unwrap()*wf.value(cfg).unwrap()
    }
}

pub struct KineticEnergy {}

impl KineticEnergy {
    pub fn  new() -> Self {
        KineticEnergy{}
    }
}

impl<T> Operator<T> for KineticEnergy
where T: Function<f64, D=Ix2>
{
    type V = f64;

    fn act_on(&self, wf: &T, cfg: &Array2<Self::V>) -> Self::V {
        1.0 // TODO implement
    }
}

pub struct IonicHamiltonian {
    v: IonicPotential,
    t: KineticEnergy
}

impl IonicHamiltonian {
    pub fn new(t: KineticEnergy, v: IonicPotential) -> Self {
        IonicHamiltonian{v, t}
    }
}

impl<T> Operator<T> for IonicHamiltonian
where T: Function<f64, D=Ix2>
{
    type V = f64;

    fn act_on(&self, wf: &T, cfg: &Array2<Self::V>) -> Self::V {
        self.t.act_on(wf, cfg) + self.v.act_on(wf, cfg)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

}