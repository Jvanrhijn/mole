// Third party imports
use ndarray::{Array1, Ix1};
// First party imports
use math::basis;
use traits::function::Function;

pub struct HydrogenGroundState {}

impl Function<f64> for HydrogenGroundState {

    type E = ();
    type D = Ix1;

    fn value(&self, cfg: &Array1<f64>) -> Result<f64, ()> {
        Ok(basis::hydrogen_1s(cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h_1s() {
        let hgs = HydrogenGroundState{};
        assert!(hgs.value(&Array1::<f64>::ones(3)).unwrap() > 0.);
    }
}