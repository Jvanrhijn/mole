// Third party imports
use ndarray::Array1;

// function returning value, gradient and laplacian
pub type Vgl = (f64, Array1<f64>, f64);
pub type Func = Fn(&Array1<f64>) -> Vgl;

fn get_spherical_angles(vector: &Array1<f64>) -> (f64, f64, f64) {
    let (x, y, z) = match vector.as_slice().expect("Empty position vector") {
        [a, b, c, ..] => (a, b, c),
        _ => unreachable!(),
    };
    let r = (vector * vector).scalar_sum().sqrt();
    let polar_angle = (z / r).acos();
    let azithmutal_angle = (y / x).atan();
    (r, polar_angle, azithmutal_angle)
}

fn radial_unit_vector(vector: &Array1<f64>) -> Array1<f64> {
    let (_, theta, phi) = get_spherical_angles(vector);
    Array1::from_vec(vec![
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    ])
}

/// Return value and laplacian of the 1s hydrogen orbital
pub fn hydrogen_1s(pos: &Array1<f64>, width: f64) -> Vgl {
    let (r, _, _) = get_spherical_angles(pos);
    let exp = (-r / width).exp();
    let value = exp;
    // convert cartesian coordinates to spherical
    let gradient = -1.0 * exp / (width * 3.0_f64.sqrt()) * radial_unit_vector(pos);
    let laplacian = (1. / width.powi(2) - 2. / (r * width)) * exp;
    (value, gradient, laplacian)
}

/// Return value and laplacian of the 2s hydrogen orbital
pub fn hydrogen_2s(pos: &Array1<f64>, width: f64) -> Vgl {
    let r = (pos * pos).scalar_sum().sqrt();
    let exp = (-r / width).exp();
    let value = (2. - 2. * r / width) * exp;
    let gradient = -2.0 * exp / width * (2.0 - r / width) * radial_unit_vector(pos);
    let laplacian = 2.0 * exp * (5. / width.powi(2) - 4. / (width * r) - r / width.powi(3));
    (value, gradient, laplacian)
}

/// Return value and Laplacian of Gaussian orbital
pub fn gaussian(pos: &Array1<f64>, width: f64) -> Vgl {
    let r = (pos * pos).scalar_sum().sqrt();
    let width2 = width.powi(2);
    let exp = (-(r).powi(2) / (2.0 * width2)).exp();
    let value = exp;
    let gradient = -r / width * exp * radial_unit_vector(pos);
    let laplacian = exp / width2 * (r.powi(2) / width2 - 3.0);
    (value, gradient, laplacian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use rand::distributions::Range;

    #[test]
    fn hydrogen_ground_state() {
        let r = Array1::<f64>::random(3, Range::new(-1., 1.));
        let wf_val = hydrogen_1s(&r, 1.0).0;
        assert!(wf_val > 0.);
    }

}
