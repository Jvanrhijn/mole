use std::result::Result::{Ok, Err};
use ndarray::{Array, ArrayBase};
use ndarray_linalg::qr::QR;

pub fn determinant<T: QR>(mat: T) -> Result<(f64), ()> {
    let (q, r) = mat.qr().unwrap();
    Ok(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_det_square() {
        assert!(false);
    }
}