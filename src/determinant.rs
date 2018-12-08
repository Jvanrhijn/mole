// Standard imports
use std::vec::Vec;
use std::result::Result;
// Third party imports
use ndarray::{Ix2, Ix1, Array, Array1, Array2};
use ndarray_linalg::{solve::Determinant, Inverse};
// First party imports
use traits::function::*;
use traits::wavefunction::WaveFunction;
use traits::differentiate::Differentiate;
use traits::cache::Cache;
use error::{Error};

pub struct Slater<T: Function<f64, D=Ix1> + Differentiate<D=Ix1>> {
    orbs: Vec<T>,
    matrix: Array2<f64>,
    matrix_laplac: Array2<f64>,
    inv_matrix: Array2<f64>,
    current_value: f64,
    current_laplac: f64

}

impl<T: Function<f64, D=Ix1> + Differentiate<D=Ix1>> Slater<T> {

    pub fn new(orbs: Vec<T>) -> Self {
        let mat_dim = orbs.len();
        let matrix = Array::<f64, Ix2>::eye(mat_dim);
        let matrix_laplac = Array::<f64, Ix2>::eye(mat_dim);
        let inv = matrix.inv().expect("Failed to take matrix inverse");
        Self{
            orbs,
            matrix: matrix,
            matrix_laplac: matrix_laplac,
            inv_matrix: inv,
            current_value: 0.0,
            current_laplac: 0.0
        }
    }

    /// Build matrix of orbitals and laplacians of orbitals
    // TODO: Build 3d-array of gradient values as well
    fn build_matrices(&self, cfg: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), Error> {
        let mat_dim = self.orbs.len();
        let mut matrix = Array2::<f64>::zeros((mat_dim, mat_dim));
        let mut matrix_laplac = Array2::<f64>::zeros((mat_dim, mat_dim));
        for i in 0..mat_dim {
            for j in 0..mat_dim {
                let slice = cfg.slice(s![i, ..]);
                let pos = array![slice[0], slice[1], slice[2]];
                matrix[[i, j]] = self.orbs[j].value(&pos)?;
                matrix_laplac[[i, j]] = self.orbs[j].laplacian(&pos)?;
            }
        }
        Ok((matrix, matrix_laplac))
    }

}

impl<T> Function<f64> for Slater<T> where T: Function<f64, D=Ix1> + Differentiate<D=Ix1> {

    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64, Error> {
        let (matrix, _) = self.build_matrices(cfg)?;
        Ok(matrix.det()?)
    }
}

impl<T> Differentiate for Slater<T>
where T: Function<f64, D=Ix1> + Differentiate<D=Ix1>
{
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Array2<f64> {
        // TODO implement
        let shape = cfg.shape();
        Array2::<f64>::ones((shape[0], shape[1]))
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64, Error> {
        // TODO implement more efficienctly
        let mat_dim = self.orbs.len();
        let (matrix, _) = self.build_matrices(cfg)?;
        let det = matrix.det()?;
        let mat_inv = matrix.inv()?;
        let mut result = 0.;
        for i in 0..mat_dim {
            let ri = array![cfg[[i, 0]], cfg[[i, 1]], cfg[[i, 2]]];
            for j in 0..mat_dim {
                result += self.orbs[j].laplacian(&ri)?*mat_inv[[j, i]];
            }
        }
        Ok(result*det)
    }

}

impl<T> WaveFunction for Slater<T>
where T: Function<f64, D=Ix1> + Differentiate<D=Ix1>
{
    fn num_electrons(&self) -> usize {
        self.orbs.len()
    }
}

impl<'a, T> Cache<Array2<f64>> for Slater<T>
where T: Function<f64, D=Ix1> + Differentiate<D=Ix1>
{
    type A = Array2<f64>;
    type V = (f64, f64);
    type U = usize;

    fn refresh(&mut self, new: &Array2<f64>) {
        //self.matrix = self.build_matrix(new).expect("Failed to construct matrix");
        let (values, laplacians) = self.build_matrices(new).expect("Failed to construct matrix");
        self.matrix = values;
        self.matrix_laplac = laplacians;
        self.inv_matrix = self.matrix.inv().expect("Failed to take matrix inverse");
        self.current_value = self.matrix.det().expect("Failed to take matrix determinant");
        self.current_laplac = self.current_value * (&self.matrix_laplac * &self.inv_matrix.t()).scalar_sum();
    }

    fn update(&self, ud: Self::U, new: &Self::A) -> (Vec<Self::A>, Self::V) {
        // determinant value: |D(x')| = |D(x)|\sum_{j=1}^N \phi_j (x_i')d_{ji}^{-1}(x)$
        let orbvec = Array1::<f64>::from_vec(self.orbs.iter().map(|phi| {
            phi.value(&new.slice(s![ud, ..]).to_owned()).expect("Failed to evaluate orbital")
        }).collect());
        let orbvec_laplac = Array1::<f64>::from_vec(self.orbs.iter().map(|phi| {
            phi.laplacian(&new.slice(s![ud, ..]).to_owned()).expect("Failed to evaluate orbital")
        }).collect());

        // compute updated wave function value
        let ratio = orbvec.dot(&self.inv_matrix.slice(s![.., ud]));
        let value = self.current_value * ratio;

        // calculate updated matrix, laplacian matrix, and inverse matrix; only need to update column `ud`
        let mut matrix = self.matrix.clone();
        let mut matrix_laplac = self.matrix_laplac.clone();
        let mut inv_matrix = self.inv_matrix.clone();
        for j in 0..self.num_electrons() {
            matrix[[ud, j]] = orbvec[j];
            matrix_laplac[[ud, j]] = orbvec_laplac[j];
        }

        // calcualte updated inverse matrix
        for j in 0..self.num_electrons() {
            if j != ud {
                let s = matrix.slice(s![ud, ..]).dot(&inv_matrix.slice(s![.., j]));
                let term = s / ratio * inv_matrix.slice(s![.., ud]).to_owned();
                let mut inv_mat_slice = inv_matrix.slice_mut(s![.., j]);
                inv_mat_slice -= &term;
            }
        }
        {
            let mut inv_mat_slice = inv_matrix.slice_mut(s![.., ud]);
            inv_mat_slice /= ratio;
        }

        // calculate new laplacian
        let current_laplac = value * (&matrix_laplac * &inv_matrix.t()).scalar_sum();

        // return updated values
        (vec![matrix, matrix_laplac, inv_matrix], (value, current_laplac))
    }

    fn set_cache(&mut self, mut storage: Vec<Self::A>, value: Self::V) {
        self.inv_matrix = storage.pop().expect("Failed to retrieve inverse matrix");
        self.matrix_laplac = storage.pop().expect("Failed to retrieve laplacian matrix");
        self.matrix = storage.pop().expect("Failed to retrieve slater matrix");
        self.current_value = value.0;
        self.current_laplac = value.1;
    }

    fn update_inplace(&mut self, ud: Self::U, new: &Array2<f64>) {
        let (new_storage, new_value) = self.update(ud, new);
        self.set_cache(new_storage, new_value);
    }

    fn current_value(&self) -> Self::V {
        (self.current_value, self.current_laplac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use orbitals::Orbital;
    use math::basis::{hydrogen_1s, hydrogen_2s};

    static EPS: f64 = 1e-15;

    #[test]
    fn value_single_electron() {
        let basis = vec![Box::new(hydrogen_1s)];
        let orb = Orbital::new(array![1.0], &basis);
        let det = Slater::new(vec![orb]);
        let x = array![[1.0, -1.0, 1.0]];
        assert_eq!(det.value(&x).unwrap(), hydrogen_1s(&array![1.0, 1.0, 1.0]).0);
    }

    #[test]
    fn value_multiple_electrons() {
        let basis: Vec<Box<Fn(&Array1<f64>) -> (f64, f64)>> = vec![
            Box::new(hydrogen_1s),
            Box::new(hydrogen_2s)
        ];
        let orbs = vec![Orbital::new(array![1.0, 0.0], &basis), Orbital::new(array![0.0, 1.0], &basis)];
        let det = Slater::new(orbs);
        let x = array![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        // manually construct orbitals
        let phi11 = hydrogen_1s(&x.slice(s![0, ..]).to_owned()).0;
        let phi22 = hydrogen_2s(&x.slice(s![1, ..]).to_owned()).0;
        let phi12 = hydrogen_1s(&x.slice(s![1, ..]).to_owned()).0;
        let phi21 = hydrogen_2s(&x.slice(s![0, ..]).to_owned()).0;
        let value = phi11*phi22 - phi21*phi12;
        assert_eq!(det.value(&x).unwrap(), value);
        // switching electron positions should flip determinant sign
        let x_switched = array![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        assert_eq!(det.value(&x_switched).unwrap(), -value);
    }

    #[test]
    fn cache() {
        let basis: Vec<Box<Fn(&Array1<f64>) -> (f64, f64)>> = vec![
            Box::new(hydrogen_1s),
            Box::new(hydrogen_2s)
        ];
        let orbsc = vec![Orbital::new(array![1.0, 0.0], &basis), Orbital::new(array![0.0, 1.0], &basis)];
        let orbs = vec![Orbital::new(array![1.0, 0.0], &basis), Orbital::new(array![0.0, 1.0], &basis)];
        let mut cached = Slater::new(orbsc);
        let not_cached = Slater::new(orbs);

        // arbitrary configuration
        let x = array![[-1.0, 0.5, 0.0], [1.0, 0.2, 1.0]];
        // initialize cache
        cached.refresh(&x);

        let (cval, clap) = cached.current_value();
        let val = not_cached.value(&x).unwrap();
        let lap = not_cached.laplacian(&x).unwrap();
        assert_eq!(cval, val);
        assert_eq!(clap,  lap);
    }

    #[test]
    fn update_cache() {
        let basis: Vec<Box<Fn(&Array1<f64>) -> (f64, f64)>> = vec![
            Box::new(hydrogen_1s),
            Box::new(hydrogen_2s)
        ];
        let orbsc = vec![Orbital::new(array![1.0, 0.0], &basis), Orbital::new(array![0.0, 1.0], &basis)];
        let orbs = vec![Orbital::new(array![1.0, 0.0], &basis), Orbital::new(array![0.0, 1.0], &basis)];
        let mut cached = Slater::new(orbsc);
        let not_cached = Slater::new(orbs);

        // arbitrary configuration
        let x = array![[-1.0, 0.5, 0.0], [1.0, 0.2, 1.0]];
        // initialize cache
        cached.refresh(&x);

        // move the first electron
        let xmov = array![[1.0, 2.0, 3.0], [1.0, 0.2, 1.0]];
        // update the cached wave function
        cached.update_inplace(0, &xmov);

        // retrieve values
        let (cval, clap) = cached.current_value();
        let val = not_cached.value(&xmov).unwrap();
        let lap = not_cached.laplacian(&xmov).unwrap();
        assert!((cval - val).abs() < EPS);
        assert!((clap - lap).abs() < EPS);
    }


}