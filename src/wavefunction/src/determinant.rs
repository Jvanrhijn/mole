// Standard imports
use std::collections::VecDeque;
use std::vec::Vec;
// Third party imports
use ndarray::{Array, Array1, Array2, Array3, Axis, Ix1, Ix2, Ix3};
use ndarray_linalg::{solve::Determinant, Inverse};
// First party imports
use errors::Error::{self, EmptyCacheError, FuncError};
use optimize::Optimize;
use wavefunction_traits::{Differentiate, Function, WaveFunction};

type Result<T> = std::result::Result<T, Error>;

type Vgl = (f64, Array2<f64>, f64);
type Ovgl = (Option<f64>, Option<Array2<f64>>, Option<f64>);
type VglMat = (Array2<f64>, Array3<f64>, Array2<f64>);

#[derive(Clone)]
pub struct Slater<T: Function<f64, D = Ix1> + Differentiate<D = Ix1>> {
    orbs: Vec<T>,
    matrix_queue: VecDeque<Array2<f64>>,
    matrix_grad_queue: VecDeque<Array3<f64>>,
    matrix_laplac_queue: VecDeque<Array2<f64>>,
    inv_matrix_queue: VecDeque<Array2<f64>>,
    current_value_queue: VecDeque<f64>,
    current_grad_queue: VecDeque<Array2<f64>>,
    current_laplac_queue: VecDeque<f64>,
}

impl<T: Function<f64, D = Ix1> + Differentiate<D = Ix1>> Slater<T> {
    pub fn new(orbs: Vec<T>) -> Result<Self> {
        let mat_dim = orbs.len();
        let matrix = Array::<f64, Ix2>::eye(mat_dim);
        let matrix_grad = Array::<f64, Ix3>::zeros((mat_dim, mat_dim, mat_dim));
        let matrix_laplac = Array::<f64, Ix2>::zeros((mat_dim, mat_dim));
        // Scale matrix for stability in inversion
        let scale = matrix.iter().fold(0.0_f64, |a, b| a.abs().max(b.abs()));
        let inv = (1.0 / scale * &matrix).inv()? / scale;
        // put cached data in queues
        let matrix_queue = VecDeque::from(vec![matrix]);
        let matrix_laplac_queue = VecDeque::from(vec![matrix_laplac]);
        let matrix_grad_queue = VecDeque::from(vec![matrix_grad]);
        let inv_matrix_queue = VecDeque::from(vec![inv]);
        let current_value_queue = VecDeque::from(vec![0.0]);
        let current_grad_queue = VecDeque::from(vec![Array2::zeros((mat_dim, 3))]);
        let current_laplac_queue = VecDeque::from(vec![0.0]);
        // construct Self
        Ok(Self {
            orbs,
            matrix_queue,
            matrix_grad_queue,
            matrix_laplac_queue,
            inv_matrix_queue,
            current_value_queue,
            current_grad_queue,
            current_laplac_queue,
        })
    }

    /// Build matrix of orbital values, gradients and laplacians
    fn build_matrices(&self, cfg: &Array2<f64>) -> Result<VglMat> {
        let mat_dim = self.orbs.len();
        let mut matrix = Array2::<f64>::zeros((mat_dim, mat_dim));
        let mut matrix_grad = Array3::<f64>::zeros((mat_dim, mat_dim, 3));
        let mut matrix_laplac = Array2::<f64>::zeros((mat_dim, mat_dim));
        for i in 0..mat_dim {
            for j in 0..mat_dim {
                let pos = cfg.slice(s![i, ..]).to_owned();
                matrix[[i, j]] = self.orbs[j].value(&pos)?;
                let orbgrad = self.orbs[j].gradient(&pos)?;
                for k in 0..3 {
                    matrix_grad[[i, j, k]] = orbgrad[k];
                }
                matrix_laplac[[i, j]] = self.orbs[j].laplacian(&pos)?;
            }
        }
        Ok((matrix, matrix_grad, matrix_laplac))
    }
}

impl<T> Function<f64> for Slater<T>
where
    T: Function<f64, D = Ix1> + Differentiate<D = Ix1>,
{
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64> {
        let (matrix, _, _) = self.build_matrices(cfg)?;
        Ok(matrix.det()?)
    }
}

impl<T> Differentiate for Slater<T>
where
    T: Function<f64, D = Ix1> + Differentiate<D = Ix1>,
{
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Result<Array2<f64>> {
        let mat_dim = self.orbs.len();
        let (matrix, matrix_grad, _) = self.build_matrices(cfg)?;
        let det = matrix.det()?;
        let mat_inv = matrix.inv()?;
        let mut result = Array2::zeros((mat_dim, 3));
        for i in 0..mat_dim {
            for j in 0..mat_dim {
                for k in 0..3 {
                    result[[i, k]] += matrix_grad[[i, j, k]] * mat_inv[[j, i]];
                }
            }
        }
        Ok(result * det)
    }

    fn laplacian(&self, cfg: &Array<f64, Self::D>) -> Result<f64> {
        let mat_dim = self.orbs.len();
        let (matrix, _, matrix_laplac) = self.build_matrices(cfg)?;
        let det = matrix.det()?;
        let mat_inv = matrix.inv()?;
        let mut result = 0.;
        for i in 0..mat_dim {
            for j in 0..mat_dim {
                result += matrix_laplac[[i, j]] * mat_inv[[j, i]];
            }
        }
        Ok(result * det)
    }
}

impl<T> WaveFunction for Slater<T>
where
    T: Function<f64, D = Ix1> + Differentiate<D = Ix1>,
{
    fn num_electrons(&self) -> usize {
        self.orbs.len()
    }
}

//impl<T> Optimize for Slater<T>
//where
//    T: Function<f64, D = Ix1> + Differentiate<D = Ix1> + Optimize,
//{
//    fn parameter_gradient(&self, cfg: &Array2<f64>) -> Result<Array1<f64>> {
//
//    }
//
//    fn num_parameters(&self) -> usize {
//        self.orbs.iter().fold(0, |n, o| o.num_parameters())
//    }
//
//    fn parameters(&self) -> &Array1<f64> {
//        // TODO: rework to avoid clone
//        &Array1::from_vec(self.orbs.iter().fold(Vec::new(), |ps, o| {
//            ps.append(&mut o.parameters().to_vec());
//            ps
//        }))
//    }
//
//    fn update_parameters(&mut self, deltap: &Array1<f64>) {
//
//    }
//
//}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbitals::Orbital;
    use basis::{hydrogen_1s, hydrogen_2s, Func, GaussianBasis, Hydrogen1sBasis};

    static EPS: f64 = 1e-15;

    #[test]
    fn value_single_electron() {
        //let basis: Vec<Box<Func>> = vec![Box::new(|x| hydrogen_1s(x, 1.0))];
        let basis = Hydrogen1sBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0]);
        let orb = Orbital::new(array![[1.0]], basis);
        let det = Slater::new(vec![orb]).unwrap();
        let x = array![[1.0, -1.0, 1.0]];

        assert!(
            (det.value(&x).unwrap() - hydrogen_1s(&x.slice(s![0, ..]).to_owned(), 1.0).0).abs()
                < 1e-15
        );
        assert!(det
            .gradient(&x)
            .unwrap()
            .slice(s![0, ..])
            .all_close(&hydrogen_1s(&x.slice(s![0, ..]).to_owned(), 1.0).1, EPS));
    }

    #[test]
    fn value_multiple_electrons() {
        let basis = Hydrogen1sBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0, 2.0]);

        let orb1 = Orbital::new(array![[1.0, 0.0]], basis.clone());
        let orb2 = Orbital::new(array![[0.0, 1.0]], basis);
        let orbs = vec![orb1, orb2];
        let det = Slater::new(orbs).unwrap();
        let x = array![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        // manually construct orbitals
        let phi11 = hydrogen_1s(&x.slice(s![0, ..]).to_owned(), 1.0).0;
        let phi22 = hydrogen_2s(&x.slice(s![1, ..]).to_owned(), 2.0).0;
        let phi12 = hydrogen_1s(&x.slice(s![1, ..]).to_owned(), 1.0).0;
        let phi21 = hydrogen_2s(&x.slice(s![0, ..]).to_owned(), 2.0).0;
        let value = phi11 * phi22 - phi21 * phi12;

        assert!((det.value(&x).unwrap() - value).abs() < 1e-15);

        // switching electron positions should flip determinant sign
        let x_switched = x.permuted_axes([0, 1]);
        assert!((det.value(&x_switched).unwrap() - (-value)).abs() < 1e-15);
    }

    #[test]
    fn gradient_and_laplacian() {
        use crate::util::grad_laplacian_finite_difference;
        let basis = GaussianBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0, 2.0, 3.0]);
        let orbitals = vec![
            Orbital::new(array![[1.0, 0.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 1.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 0.0, 1.0]], basis.clone()),
        ];
        let det = Slater::new(orbitals).unwrap();
        let cfg = array![[1.0, 1.0, 3.0], [0.5, 0.02, -0.8], [1.1, -0.5, 0.2]];
        let (grad_fd, laplac_fd) = grad_laplacian_finite_difference(&det, &cfg, 1e-5).unwrap();

        assert!(det.gradient(&cfg).unwrap().all_close(&grad_fd, 1e-5));
        assert!((det.laplacian(&cfg).unwrap() - laplac_fd).abs() < 1e-5);
    }

    #[test]
    fn cache() {
        let basis = Hydrogen1sBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0, 0.5]);

        let orbsc = vec![
            Orbital::new(array![[1.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 1.0]], basis.clone()),
        ];
        let orbs = vec![
            Orbital::new(array![[1.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 1.0]], basis),
        ];
        let mut cached = Slater::new(orbsc).unwrap();
        let not_cached = Slater::new(orbs).unwrap();

        // arbitrary configuration
        let x = array![[-1.0, 0.5, 0.0], [1.0, 0.2, 1.0]];
        // initialize cache
        cached.refresh(&x).unwrap();

        let (cval, cgrad, clap) = cached.current_value().unwrap();
        let val = not_cached.value(&x).unwrap();
        let grad = not_cached.gradient(&x).unwrap();
        let lap = not_cached.laplacian(&x).unwrap();

        assert!((cval - val).abs() < 1e-10);
        assert!(grad.all_close(&cgrad, EPS));
        assert!((clap - lap).abs() < 1e-10);
    }

    #[test]
    fn update_cache() {
        let _basis: Vec<Box<Func>> = vec![
            Box::new(|x| hydrogen_1s(&x, 1.0)),
            Box::new(|x| hydrogen_2s(&x, 0.5)),
        ];
        let basis = Hydrogen1sBasis::new(array![[0.0, 0.0, 0.0]], vec![1.0, 0.5]);
        let orbsc = vec![
            Orbital::new(array![[1.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 1.0]], basis.clone()),
        ];
        let orbs = vec![
            Orbital::new(array![[1.0, 0.0]], basis.clone()),
            Orbital::new(array![[0.0, 1.0]], basis),
        ];
        let mut cached = Slater::new(orbsc).unwrap();
        let not_cached = Slater::new(orbs).unwrap();

        // arbitrary configuration
        let x = array![[-1.0, 0.5, 0.0], [1.0, 0.2, 1.0]];
        // initialize cache
        cached.refresh(&x).unwrap();

        // move the first electron
        let xmov = array![[1.0, 2.0, 3.0], [1.0, 0.2, 1.0]];
        // update the cached wave function
        cached.enqueue_update(0, &xmov).unwrap();
        cached.push_update();

        // retrieve values
        let (cval, cgrad, clap) = cached.current_value().unwrap();
        let val = not_cached.value(&xmov).unwrap();
        let grad = not_cached.gradient(&xmov).unwrap();
        let lap = not_cached.laplacian(&xmov).unwrap();

        assert!((cval - val).abs() < EPS);
        assert!(cgrad.all_close(&grad, EPS));
        assert!((clap - lap).abs() < EPS);
    }
}
