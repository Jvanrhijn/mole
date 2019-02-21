use std::vec::Vec;
use std::io::{self, BufRead, Write, Result};
use std::fs::File;


/// Read a stream of data from stdin line by line
fn read_data() -> Result<Vec<f64>> {
    let stdin = io::stdin();
    let data = stdin.lock().lines()
        .map(|x| x.unwrap().split_whitespace().collect::<Vec<_>>()[1]
             .parse::<f64>().expect("Failed to parse as float"))
        .collect::<Vec<_>>();
    Ok(data)
}


/// Compute the mean of an array of floats
fn mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>()/(data.len() as f64)
}


/// Compute the variance of an array of floats, with given DOF
fn variance(data: &[f64], ddof: usize) -> f64 {
    let average = mean(data);
    data.iter().map(|x| (average - x).powi(2)).sum::<f64>()/(data.len() -ddof) as f64
}


/// Compute the correlation time, effective steps, and sigma of an array of data
fn correlation(data: &[f64], mean: f64, variance: f64) -> Result<(f64, f64, f64)> {
    const MAX_STEPS: usize = 200;
    let nsteps = data.len();
    let mut tcorr: f64 = 1.0;
    let max_i = MAX_STEPS.min(nsteps - 1);

    let mut file = File::create("corr.out")?;
    let mut f = 1.0;

    for i in 1..=max_i {
        let mut corr = data[..nsteps-i].iter().zip(data[i..].iter())
            .map(|(&x, &xshift)| (x - mean)*(xshift - mean))
            .take(nsteps - i)
            .sum::<f64>();
        corr /= variance*(nsteps - i) as f64;
        if corr < 0. {
            f = 0.0;
        }
        tcorr += 2.0 * corr * f;
        file.write(format!("{} {}\n", i, corr).as_bytes())?;
    }
    tcorr = tcorr.max(1.0);
    Ok((tcorr, nsteps as f64 / tcorr, (variance*tcorr / nsteps as f64).sqrt()))
}


/// Perform blocking analysis over data
fn blocking(data: &[f64]) -> Result<()> {
    let ndata = data.len();
    const MIN_LEFT: usize = 20;
    const NSIZES: usize = 100;
    let large = ndata / MIN_LEFT;
    let step_size = (large/NSIZES).max(1);
    
    let mut file = File::create("blocking.out")?;

    for size in (1..=large).step_by(step_size) {
        let nblocks = ndata/size;
        // block averages
        let blocks = data.chunks(size);
        let ave_blks = blocks.map(mean);
        let ave_sq = mean(&ave_blks.clone().map(|x| x.powi(2)).collect::<Vec<_>>());
        let ave = mean(&ave_blks.collect::<Vec<_>>());
        // estimated error at this block size
        let error = ((ave_sq - ave.powi(2))/(nblocks - 1) as f64).sqrt();
        // write to file
        file.write(format!("{} {}\n", size, error).as_bytes())?;
    }

    Ok(())
}


fn main() {
    let data = read_data().expect("Failed to parse input stream");
    let data = &data[..data.len()-1];
    let average = mean(&data);
    let var = variance(&data, 1);
    let (tcorr, neff, sigma) = correlation(&data, average, var)
        .expect("Failed to write correlation to file");

    blocking(&data).expect("Failed to write blocking data to file");

    println!("average\t\t{}", average);
    println!("variance\t{}", var);
    println!("tcorr\t\t{}", tcorr);
    println!("n eff\t\t{}", neff);
    println!("sigma\t\t{}", sigma);
}
