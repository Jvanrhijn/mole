// Standard imports
use std::result::Result::{Ok, Err};
// Third pary imports
use assert;
use ndarray::{Array2, Array};
use ndarray_linalg::{qr::{QR, QRSquare, QRSquareInto}, error::LinalgError};
use ndarray_rand::RandomExt;
use rand::distributions::Range;

pub fn det_abs<M>(mat: &M) -> Result<f64, LinalgError> 
where
    M: QRSquare<R=Array2<f64>>,
{
    //println!("{:?}", mat);
    let (_, r) = mat.qr_square()?;
    Ok(r.diag().iter().fold(1., |acc, x| acc*x).abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_det_square() {
        //let mat: Array2<f64> = array![[1., 2.], [3., 4.]];
        let mat: Array2<f64> = Array::random((2, 2), Range::new(-1., 1.));
        let det: f64 = mat[[0, 0]]*mat[[1, 1]] - mat[[0, 1]]*mat[[1, 0]];
        assert::close(det_abs(&mat).unwrap(), det.abs(), 1e-5);
    }
}
