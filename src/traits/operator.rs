use ndarray::{Array2};

use traits::function::Function;
use error::{Error};

pub trait Operator<T>
where T: Function<Self::V> + ?Sized
{
    type V;

    fn act_on(&self, wf: &T, cfg: &Array2<Self::V>) -> Result<Self::V, Error>;
}
