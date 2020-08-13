use mole::prelude::*;
use ndarray::{s, Axis, Ix2, Array1, Array2, array};
use ndarray_linalg::Norm;
use ndarray_rand::RandomExt;
use rand::{SeedableRng, StdRng};
use itertools::izip;
use rand::distributions::{Normal, Range};
use rand::{FromEntropy, Rng};

use rgsl::types::mathieu::MathieuWorkspace;
use num::complex::Complex;

use std::collections::HashMap;

// Create a very basic logger
struct Logger;
impl Log for Logger {
    fn log(&mut self, data: &HashMap<String, Vec<OperatorValue>>) -> String {
        format!(
            "Energy: {}",
            data.get("Energy")
                .unwrap()
                .iter()
                .last()
                .unwrap()
                .get_scalar()
                .unwrap(),
        )
    }
}


#[derive(Clone, Debug)]
struct TrialWaveFunc {
    a: f64,
    q: f64,
}

impl TrialWaveFunc {
    pub fn new(a: f64) -> Self {
        Self {
            q: 0.8253525490491695,
            a: a,
        }
    }
}

impl Function<f64> for TrialWaveFunc {
    type D = Ix2;

    fn value(&self, cfg: &Array2<f64>) -> Result<f64> {
        let x = cfg[[0, 0]];
        let y = cfg[[0, 1]];
        let combined = (Complex::new(x, y) / self.a).acosh();
        let mu = combined.re;
        let nu = combined.im;

        let resmu = MathieuWorkspace::mathieu_Mc(1, 0, self.q, mu).1.val;
        let resnu = MathieuWorkspace::mathieu_ce(0, self.q, nu).1.val;

        Ok(if mu <= 1.0 { x * resmu * resnu } else { 0.0 })
    }
}

impl Differentiate for TrialWaveFunc {
    type D = Ix2;

    fn gradient(&self, cfg: &Array2<f64>) -> Result<Array2<f64>> {
        const DX: f64 = 1e-5;
        let dx = array![[DX, 0.0]];
        let dy = array![[0.0, DX]];
        let mut grad = Array2::zeros((1, 2));
        grad[[0, 0]] = (self.value(&(cfg + &dx))? - self.value(&(cfg - &dx))?) / (2.0*DX);
        grad[[0, 1]] = (self.value(&(cfg + &dy))? - self.value(&(cfg - &dy))?) / (2.0*DX);
        Ok(grad)
    }

    fn laplacian(&self, cfg: &Array2<f64>) -> Result<f64> {
        const DX: f64 = 1e-5;
        let dx = array![[DX, 0.0]];
        let dy = array![[0.0, DX]];
        let val = self.value(cfg)?;
        let res = (self.value(&(cfg + &dx))? - 2.0*val + self.value(&(cfg - &dx))?) / DX.powi(2)
                + (self.value(&(cfg + &dy))? - 2.0*val + self.value(&(cfg - &dy))?) / DX.powi(2);
        Ok(res)
    }
}

impl WaveFunction for TrialWaveFunc {
    fn num_electrons(&self) -> usize {
        1
    }

    fn dimension(&self) -> usize {
        2
    }
}

struct EnergySec {
    wf: TrialWaveFunc,
    ke: KineticEnergy,
}

struct Transformation;
impl Transformation {
    pub fn node_dist(wf: &TrialWaveFunc, cfg: &Array2<f64>) -> Result<f64> {
        Ok(wf.value(cfg)? / wf.gradient(cfg)?.norm_l2())
    }

    pub fn normal(wf: &TrialWaveFunc, cfg: &Array2<f64>) -> Result<Array2<f64>> {
        let grad = wf.gradient(cfg)?;
        Ok(1.0 / grad.norm_l2() * grad)
    }

    pub fn cutoff(d: f64) -> (f64, f64) {
        const a: f64 = 0.025;
        const b: f64 = 1e-2;
        let value = 0.5 * (1.0 + f64::tanh((a - d) / b));
        let deriv = if d > 0.1 { 0.0 } else { -1.0 / (2.0*b) / f64::cosh((a - d)/b).powi(2) };
        (value, deriv)
    }
}

impl EnergySec {
    pub fn new(wf: TrialWaveFunc) -> Self {
        Self {
            wf,
            ke: KineticEnergy::new()
        }
    }
}

impl LocalOperator<TrialWaveFunc> for EnergySec {
    fn act_on(&self, wf_primary: &TrialWaveFunc, cfg: &Array2<f64>) -> Result<OperatorValue> {
            // perform NW transform
            //let nprime = Transformation::normal(&self.wf, cfg)?;
            //let dist = Transformation::node_dist(wf_primary, cfg)?;
            //let dist_prime = Transformation::node_dist(&self.wf, cfg)?;
            //let (u, _) = Transformation::cutoff(dist);
            //let psi_prime = self.wf.value(cfg)?;
            //let cfg_warp = cfg + &((dist - dist_prime)*f64::signum(psi_prime)*u*&nprime);

            self.ke.act_on(&self.wf, &cfg)
    }
}

struct Distribution {
    wf: TrialWaveFunc,
}

impl Distribution {
    pub fn new(wf: TrialWaveFunc) -> Self {
        Self { wf }
    }
}

impl LocalOperator<TrialWaveFunc> for Distribution {
    fn act_on(&self, wf_primary: &TrialWaveFunc, cfg: &Array2<f64>) -> Result<OperatorValue> {
        // perform NW transform
        //let n = Transformation::normal(wf_primary, cfg)?;
        //let nprime = Transformation::normal(&self.wf, cfg)?;
        //let dist = Transformation::node_dist(wf_primary, cfg)?;
        //let dist_prime = Transformation::node_dist(&self.wf, cfg)?;
        //let (u, uprime) = Transformation::cutoff(dist);
        //let psi = wf_primary.value(cfg)?;
        //let psi_prime = self.wf.value(cfg)?;
        //let cfg_warp = cfg + &((dist - dist_prime)*f64::signum(psi_prime)*u*&nprime);

        //let jac = 1.0 - u*f64::signum(psi_prime) - f64::signum(psi*psi_prime)   
        //    *(&n * &nprime).sum()*(u - (dist - dist_prime)*uprime);
        let jac = 1.0;

        Ok(OperatorValue::Scalar(
            self.wf.value(&cfg)?.powi(2) * jac
        ))
    }
}

fn main() {
    let a = 1.0;
    let da = 1e-5;

    let wf = TrialWaveFunc::new(a);
    let wf_sec = TrialWaveFunc::new(a + da);
    let wf_tert = TrialWaveFunc::new(a - da);

    let metrop = MetropolisBox::from_rng(0.5, StdRng::from_seed([0; 32]));
    //let metrop = MetropolisDiffuse::from_rng(0.001, StdRng::from_seed([0; 32]));

    let ke = KineticEnergy::new();
    let ke_sec = EnergySec::new(wf_sec.clone());
    let ke_tert = EnergySec::new(wf_tert.clone());

    let obs = operators! {
        "Energy" => ke,
        "Energy sec" => ke_sec,
        "Energy tert" => ke_tert,
        "Distr sec" => Distribution::new(wf_sec.clone()),
        "Distr tert" => Distribution::new(wf_tert.clone())
    };

    // pick point close to node
    //let nu_rand = rng.gen::<f64>();
    //let xstart = array![[-1.5634058306958694, -0.39321091987790857]];
    let xstart = array![[1.0, 0.0]];

    let sampler = Sampler::with_initial_configuration(wf, metrop, &obs, xstart)
       .expect("Bad initial configuration");

     // Perform the MC integration
    let runner = Runner::new(sampler, Logger);

    const NSAMPLES: usize = 10_000;
    const BLOCK_SIZE: usize = 100;
    let result = runner.run(NSAMPLES, BLOCK_SIZE).unwrap();

    let energy_data = Array1::<f64>::from_vec(
        result
            .data
            .get("Energy")
            .unwrap()
            .iter()
            .map(|x| *x.get_scalar().unwrap())
            .collect::<Vec<_>>(),
    );
    let energy_data_sec = Array1::<f64>::from_vec(
        result
            .data
            .get("Energy sec")
            .unwrap()
            .iter()
            .map(|x| *x.get_scalar().unwrap())
            .collect::<Vec<_>>(),
    );    
    let energy_data_tert = Array1::<f64>::from_vec(
        result
            .data
            .get("Energy tert")
            .unwrap()
            .iter()
            .map(|x| *x.get_scalar().unwrap())
            .collect::<Vec<_>>(),
    );
    let distr_data_sec = Array1::<f64>::from_vec(
        result
            .data
            .get("Distr sec")
            .unwrap()
            .iter()
            .map(|x| *x.get_scalar().unwrap())
            .collect::<Vec<_>>(),
    );
    let distr_data_tert = Array1::<f64>::from_vec(
        result
            .data
            .get("Distr tert")
            .unwrap()
            .iter()
            .map(|x| *x.get_scalar().unwrap())
            .collect::<Vec<_>>(),
    );

    let energy = *energy_data.mean_axis(Axis(0)).first().unwrap();

    // compute force
    let force_local: Vec<_> = izip!(&energy_data, &energy_data_sec, &energy_data_tert, &distr_data_sec, &distr_data_tert)
        .map(|(e, es, et, pis, pit)| {
            -((es - et)/(2.0*da) + (e - energy) * (pis.ln() - pit.ln()) / (2.0*da))
        }).collect();

    let force = force_local.iter().sum::<f64>() / force_local.len() as f64;
    let force_err = block_error(&force_local, BLOCK_SIZE);

    // Retrieve mean values of energy over run
    let energy = *energy_data.mean_axis(Axis(0)).first().unwrap();
    let energy_err = block_error(&energy_data.to_vec(), BLOCK_SIZE);

    println!("\nEnergy: {:.5} +/- {:.5}", energy, energy_err);
    println!("\nForce:  {:.5} +/- {:.5}", force, force_err);

}


fn block_error(data: &Vec<f64>, block_size: usize) -> f64 {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let blocks = data.chunks(block_size);
    let num_blocks = blocks.len();
    let bmeans = blocks.map(|x| x.iter().fold(0.0, |acc, x| acc + x) / x.len() as f64);
    let bmean_sq = bmeans.map(|x| x.powi(2)).sum::<f64>() / num_blocks as f64;
    f64::sqrt((bmean_sq - mean.powi(2)) / num_blocks as f64)
}
