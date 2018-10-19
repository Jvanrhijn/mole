use ndarray::{Array2};

use traits::function::Function;
use error::{Error};

pub trait Operator<T>
where T: Function<f64> + ?Sized
{
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<f64, Error>;
}
