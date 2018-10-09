// Standard imports
use std::result::Result::{Ok, Err};
// Third pary imports
use assert;
use ndarray::{Array2, Array};
use ndarray_linalg::{qr, error::LinalgError};
use ndarray_rand::RandomExt;
use rand::distributions::Range;

pub fn det_abs<M>(mat: &M) -> Result<f64, LinalgError> 
where
    M: qr::QRSquare<R=Array2<f64>>,
{
    let (_, r) = mat.qr_square()?;
    Ok(r.diag().iter().fold(1., |acc, x| acc*x).abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Neg;
    use rand::{random, distributions::uniform::SampleUniform};

    struct TestData {
        num_tests: usize
    }

    impl TestData {
        pub fn new(num_tests: usize) -> Self {
            TestData{num_tests}
        }

        pub fn random_square_matrix<T>(size: usize, range: T) -> Array2<T>
        where
            T: SampleUniform + Neg<Output=T> + Copy
        {
            Array::random((size, size), Range::new(-range, range))
        }

        pub fn random_sized_square_matrix<T>(range: T) -> Array2<T>
        where
            T: SampleUniform + Neg<Output=T> + Copy
        {
            let size = random::<u8>();
            Self::random_square_matrix(size as usize, range)
        }


    }

    fn setup() -> TestData {
        TestData::new(10)
    }

    #[test]
    fn test_det_2x2() {
        for _i in 0..setup().num_tests {
            let mat: Array2<f64> = TestData::random_square_matrix::<f64>(2, 1.);
            let det: f64 = mat[[0, 0]]*mat[[1, 1]] - mat[[0, 1]]*mat[[1, 0]];
            assert::close(det_abs(&mat).unwrap(), det.abs(), 1e-5);
        }
    }

    #[test]
    fn test_transpose() {
        for _i in 0..setup().num_tests {
            let mat: Array2<f64> = TestData::random_sized_square_matrix(1.);
            let det = det_abs(&mat).unwrap();
            let tol = det*1e-5;
            assert::close(det, det_abs(&mat.t()).unwrap(), tol);
        }
    }

}
