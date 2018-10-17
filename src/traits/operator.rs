use ndarray::{Array, Array2};

use traits::function::Function;
use error::{Error, FuncError};

pub trait Operator<T>
where T: Function<Self::V> + ?Sized
{
    type V;

    fn act_on(&self, wf: &T, cfg: &Array2<Self::V>) -> Result<Self::V, Error>;
}
