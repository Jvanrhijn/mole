// Third party imports
use ndarray::{Array, Array1, Array2, Ix1};
// First party imports
use basis::BasisSet;
use errors::Error;
use wavefunction_traits::{Differentiate, Function};
use optimize::Optimize;

/// Parametrized orbital as a linear combination of basis functions:
/// $\phi(x) = \sum_{i=1}^{N_{\text{basis}}} \xi_i(x)$.
#[derive(Clone)]
pub struct Orbital<T>
where
    T: BasisSet, 
{
    parms: Array1<f64>,
    basis_set: T,
    // TODO: rework this hack
    ncenters: usize,
    nwidths: usize,
}

impl<T> Orbital<T>
where
    T: BasisSet, 
{
    pub fn new(parms: Array2<f64>, basis_set: T) -> Self {
        let (ncenters, nwidths) = match parms.shape() {
            [ncenters, nwidths] => (*ncenters, *nwidths),
            _ => unreachable!()
        };
        Self { parms: parms.into_shape(ncenters*nwidths).unwrap(), basis_set, ncenters, nwidths }
    }
}

impl<T> Function<f64> for Orbital<T>
where
    T: BasisSet, 
{
    type D = Ix1;

    fn value(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        let parms = Array2::from_shape_vec((self.ncenters, self.nwidths), self.parms.to_vec()).unwrap();
        Ok(self.basis_set.linear_combination(cfg, &parms).0)
    }
}

impl<T> Differentiate for Orbital<T>
where
    T: BasisSet, 
{
    type D = Ix1;

    fn gradient(&self, cfg: &Array<f64, Self::D>) -> Result<Array<f64, Self::D>, Error> {
        let parms = Array2::from_shape_vec((self.ncenters, self.nwidths), self.parms.to_vec()).unwrap();
        Ok(self.basis_set.linear_combination(cfg, &parms).1)
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        let parms = Array2::from_shape_vec((self.ncenters, self.nwidths), self.parms.to_vec()).unwrap();
        Ok(self.basis_set.linear_combination(cfg, &parms).2)
    }
}

impl<T> Optimize for Orbital<T> 
where
    T: BasisSet,
{
    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>, Error> {
        let parms = Array2::from_shape_vec((self.ncenters, self.nwidths), self.parms.to_vec()).unwrap();
        Ok(self.basis_set.coefficient_derivative(&cfg.clone().into_shape(3)?, &parms).into_shape(self.ncenters*self.nwidths)?)
    }

    fn num_parameters(&self) -> usize {
        self.parms.len()
    }

    fn update_parameters(&mut self, deltap: &Array1<f64>) {
        self.parms += deltap;
    }

    fn parameters(&self) -> &Array1<f64> {
        &self.parms
    }
}