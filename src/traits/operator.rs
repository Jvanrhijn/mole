use wavefunction;

pub trait Operator {
   fn act_on<T: WaveFunction>(wf: &mut T) -> &T;
}