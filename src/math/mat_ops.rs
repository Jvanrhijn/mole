// Standard imports
// Third pary imports
use ndarray::{Array2};
use ndarray_linalg::{qr, error::LinalgError};

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
    use assert;
    use rand::{random, distributions::uniform::SampleUniform, distributions::Range};
    use ndarray_rand::RandomExt;
    use ndarray::Array;

    struct TestData {
        num_tests: usize,
        tol: f64
    }

    impl TestData {
        pub fn new(num_tests: usize, tol: f64) -> Self {
            TestData{num_tests, tol}
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
        TestData::new(10, 1e-5)
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
            let tol = det*setup().tol;
            assert::close(det, det_abs(&mat.t()).unwrap(), tol);
        }
    }

    #[test]
    fn test_product() {
        for _i in 0..setup().num_tests {
            let mat_1: Array2<f64> = TestData::random_square_matrix(5, 1.);
            let mat_2: Array2<f64> = TestData::random_square_matrix(5 , 1.);
            let prod = mat_1.dot(&mat_2);
            let det_1 = det_abs(&mat_1).unwrap();
            let det_2 = det_abs(&mat_2).unwrap();
            let tol = det_1*det_2*setup().tol;
            assert::close(det_1*det_2, det_abs(&prod).unwrap(), tol);
        }
    }

}
