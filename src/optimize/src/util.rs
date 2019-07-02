use ndarray::{Array1, Array2, Axis};
use operator::OperatorValue;
use std::collections::HashMap;

pub fn compute_energy_gradient(
    mc_data: &HashMap<String, Vec<OperatorValue>>,
    averages: &HashMap<String, OperatorValue>,
) -> Array1<f64> {
    let wf_values = mc_data
        .get("Wavefunction value")
        .unwrap()
        .iter()
        .map(|x| x.get_scalar().unwrap());
    let wf_grad = mc_data
        .get("Parameter gradient")
        .unwrap()
        .iter()
        .map(|x| x.get_vector().unwrap())
        .collect::<Vec<_>>();
    let energies = mc_data
        .get("Energy")
        .unwrap()
        .iter()
        .map(|x| x.get_scalar().unwrap());

    let energy = averages.get("Energy").unwrap().get_scalar().unwrap();

    let nparm = wf_grad[0].len();
    let nsamples = wf_values.len();

    let mut local_gradient = Array2::zeros((nsamples, nparm));

    for (mut row, psi, psi_i, el) in
        izip!(local_gradient.genrows_mut(), wf_values, wf_grad, energies)
    {
        row += &(2.0 * &((psi_i / *psi) * (el - energy)));
    }

    local_gradient.mean_axis(Axis(0))
}
