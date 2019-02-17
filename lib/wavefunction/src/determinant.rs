// Standard imports
use std::vec::Vec;
use std::collections::VecDeque;
use std::result::Result;
// Third party imports
use ndarray::{Ix2, Ix1, Array, Array1, Array2};
use ndarray_linalg::{solve::Determinant, Inverse};
// First party imports
use crate::traits::{WaveFunction, Differentiate, Cache, Function};
use crate::error::Error;

pub struct Slater<T: Function<f64, D=Ix1> + Differentiate<D=Ix1>> {
    orbs: Vec<T>,
    matrix_queue: VecDeque<Array2<f64>>,
    matrix_laplac_queue: VecDeque<Array2<f64>>,
    inv_matrix_queue: VecDeque<Array2<f64>>,
    current_value_queue: VecDeque<f64>,
    current_laplac_queue: VecDeque<f64>,
}

impl<T: Function<f64, D=Ix1> + Differentiate<D=Ix1>> Slater<T> {

    pub fn new(orbs: Vec<T>) -> Self {
        let mat_dim = orbs.len();
        let matrix = Array::<f64, Ix2>::eye(mat_dim);
        let matrix_laplac = Array::<f64, Ix2>::eye(mat_dim);
        let inv = matrix.inv().expect("Failed to take matrix inverse");
        // put cached data in queues
        let matrix_queue = VecDeque::from(vec![matrix]);
        let matrix_laplac_queue = VecDeque::from(vec![matrix_laplac]);
        let inv_matrix_queue = VecDeque::from(vec![inv]);
        let current_value_queue = VecDeque::from(vec![0.0]);
        let current_laplac_queue = VecDeque::from(vec![0.0]);
        Self{
            orbs,
            matrix_queue,
            matrix_laplac_queue,
            inv_matrix_queue,
            current_value_queue,
            current_laplac_queue
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
        let (values, laplacians) = self.build_matrices(new)
            .expect("Failed to construct matrix");
        let inv = values.inv()
            .expect("Failed to take matrix inverse");
        let value = values.det()
            .expect("Failed to take matrix determinant");

        *self.current_value_queue.front_mut().unwrap() = value;
        *self.current_laplac_queue.front_mut().unwrap() = value * (&laplacians * &inv.t()).scalar_sum();

        for (queue, data) in vec![&mut self.matrix_queue, &mut self.matrix_laplac_queue, &mut self.inv_matrix_queue].iter_mut()
            .zip(vec![values, laplacians, inv].into_iter()) {
            *queue.front_mut().expect("Attempt to retrieve data from empty queue") = data;
        }
    }

    fn enqueue_update(&mut self, ud: Self::U, new: &Self::A) {
        // TODO: refactor into smaller functions
        // determinant value: |D(x')| = |D(x)|\sum_{j=1}^N \phi_j (x_i')d_{ji}^{-1}(x)$
        let data: Vec<(f64, f64)> = self.orbs.iter().map(|phi| {
            (phi.value(&new.slice(s![ud, ..]).to_owned()).unwrap(),
             phi.laplacian(&new.slice(s![ud, ..]).to_owned()).unwrap())
        }).collect();
        let orbvec = Array1::<f64>::from_vec(data.iter().map(|x| x.0).collect());
        let orbvec_laplac = Array1::<f64>::from_vec(data.into_iter().map(|x| x.1).collect());

        // compute updated wave function value
        let ratio = orbvec.dot(&self.inv_matrix_queue.front().unwrap().slice(s![.., ud]));
        let value = self.current_value_queue.front().unwrap() * ratio;

        // calculate updated matrix, laplacian matrix, and inverse matrix; only need to update column `ud`
        let mut matrix = self.matrix_queue.front().unwrap().clone();
        let mut matrix_laplac = self.matrix_laplac_queue.front().unwrap().clone();
        let mut inv_matrix = self.inv_matrix_queue.front().unwrap().clone();
        for j in 0..self.num_electrons() {
            matrix[[ud, j]] = orbvec[j];
            matrix_laplac[[ud, j]] = orbvec_laplac[j];
        }

        // calculate updated inverse matrix
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

        // enqueue the update data
        self.matrix_queue.push_back(matrix);
        self.matrix_laplac_queue.push_back(matrix_laplac);
        self.inv_matrix_queue.push_back(inv_matrix);
        self.current_value_queue.push_back(value);
        self.current_laplac_queue.push_back(current_laplac);
    }

    fn push_update(&mut self) {
        for q in vec![&mut self.matrix_queue, &mut self.matrix_laplac_queue, &mut self.inv_matrix_queue].iter_mut() {
            q.pop_front();
        }
        for q in vec![&mut self.current_value_queue, &mut self.current_laplac_queue].iter_mut() {
            q.pop_front();
        }
    }

    fn flush_update(&mut self) {
        for q in vec![&mut self.matrix_queue, &mut self.matrix_laplac_queue, &mut self.inv_matrix_queue].iter_mut() {
            if q.len() == 2 {
                q.pop_back();
            }
        }
        for q in vec![&mut self.current_value_queue, &mut self.current_laplac_queue].iter_mut() {
            if q.len() == 2 {
                q.pop_back();
            }
        }
    }

    fn current_value(&self) -> Self::V {
        (*self.current_value_queue.front().unwrap(), *self.current_laplac_queue.front().unwrap())
    }

    fn enqueued_value(&self) -> Option<Self::V> {
        match (self.current_value_queue.back(), self.current_laplac_queue.back()) {
            (Some(&v), Some(&l)) => Some((v, l)),
            _ => None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use crate::orbitals::Orbital;
    use basis::{hydrogen_1s, hydrogen_2s, Func};

    static EPS: f64 = 1e-15;

    #[test]
    fn value_single_electron() {
        let basis: Vec<Box<Func>> = vec![Box::new(|x| hydrogen_1s(x, 1.0))];
        let orb = Orbital::new(array![1.0], &basis);
        let det = Slater::new(vec![orb]);
        let x = array![[1.0, -1.0, 1.0]];
        assert_eq!(det.value(&x).unwrap(), hydrogen_1s(&array![1.0, 1.0, 1.0], 1.0).0);
    }

    #[test]
    fn value_multiple_electrons() {
        let basis: Vec<Box<Fn(&Array1<f64>) -> (f64, f64)>> = vec![
            Box::new(|x| hydrogen_1s(&x, 1.0)),
            Box::new(|x| hydrogen_2s(&x, 2.0))
        ];
        let orbs = vec![Orbital::new(array![1.0, 0.0], &basis), Orbital::new(array![0.0, 1.0], &basis)];
        let det = Slater::new(orbs);
        let x = array![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        // manually construct orbitals
        let phi11 = hydrogen_1s(&x.slice(s![0, ..]).to_owned(), 1.0).0;
        let phi22 = hydrogen_2s(&x.slice(s![1, ..]).to_owned(), 2.0).0;
        let phi12 = hydrogen_1s(&x.slice(s![1, ..]).to_owned(), 1.0).0;
        let phi21 = hydrogen_2s(&x.slice(s![0, ..]).to_owned(), 2.0).0;
        let value = phi11*phi22 - phi21*phi12;
        assert_eq!(det.value(&x).unwrap(), value);
        // switching electron positions should flip determinant sign
        let x_switched = array![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        assert_eq!(det.value(&x_switched).unwrap(), -value);
    }

    #[test]
    fn cache() {
        let basis: Vec<Box<Fn(&Array1<f64>) -> (f64, f64)>> = vec![
            Box::new(|x| hydrogen_1s(&x, 1.0)),
            Box::new(|x| hydrogen_2s(&x, 0.5))
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
            Box::new(|x| hydrogen_1s(&x, 1.0)),
            Box::new(|x| hydrogen_2s(&x, 0.5))
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
        cached.enqueue_update(0, &xmov);
        cached.push_update();

        // retrieve values
        let (cval, clap) = cached.current_value();
        let val = not_cached.value(&xmov).unwrap();
        let lap = not_cached.laplacian(&xmov).unwrap();
        assert!((cval - val).abs() < EPS);
        assert!((clap - lap).abs() < EPS);
    }


}