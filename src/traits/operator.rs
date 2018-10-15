use ndarray::{Array, Array2};

use traits::function::Function;

pub trait Operator<T>
where T: Function<Self::V>
{

    type V;

    fn act_on(&self, wf: &T, cfg: &Array2<Self::V>) -> Self::V;
}
