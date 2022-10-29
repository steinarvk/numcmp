use clap::Parser;
use rand::Rng;
use std::cmp::Ordering;
use std::fs::File;
use std::io::BufRead;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "numcmp")]
#[command(about = "Compare two numeric samples using bootstrapping and simulation")]
struct Cli {
    /// File with baseline numbers
    #[arg(value_name = "BASELINE")]
    baseline_filename: PathBuf,

    /// File with numbers under test
    #[arg(value_name = "TARGET")]
    target_filename: PathBuf,

    /// Number of simulation iterations
    #[arg(short = 'i', long = "iterations", default_value = "10000")]
    iterations: i32,
}

#[derive(Debug)]
enum Error {
    Oops(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::Oops(e.to_string())
    }
}

impl From<std::num::ParseFloatError> for Error {
    fn from(e: std::num::ParseFloatError) -> Error {
        Error::Oops(e.to_string())
    }
}

fn read_numbers(path: std::path::PathBuf) -> Result<Vec<f64>, Error> {
    let mut rv = Vec::new();
    for line in std::io::BufReader::new(File::open(path)?).lines() {
        let x = line?.parse()?;
        rv.push(x);
    }
    Ok(rv)
}

fn read_and_sort_numbers(path: std::path::PathBuf) -> Result<Vec<f64>, Error> {
    let mut rv = read_numbers(path)?;
    rv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(rv)
}

fn is_sorted(xs: &Vec<f64>) -> bool {
    for window in xs.windows(2) {
        if window[0] > window[1] {
            return false;
        }
    }
    true
}

fn quantile_index(n: usize, q: f64) -> f64 {
    // 2 items, quantile 0.5: index should be 0.5
    // 3 items, quantile 1: index should be 1

    ((n - 1) as f64) * q
}

fn get_quantile(sorted_numbers: &Vec<f64>, q: f64) -> Result<f64, Error> {
    if sorted_numbers.is_empty() {
        return Err(Error::Oops("vector is empty".to_string()));
    }

    if q < 0.0 || q > 1.0 {
        return Err(Error::Oops(format!(
            "quantile parameter q={} is out of range [0,1]",
            q
        )));
    }

    debug_assert!(is_sorted(sorted_numbers));

    if q == 0.0 {
        return Ok(*sorted_numbers
            .first()
            .expect("vector was checked to be nonempty"));
    }
    if q == 1.0 {
        return Ok(*sorted_numbers
            .last()
            .expect("vector was checked to be nonempty"));
    }

    let qi = quantile_index(sorted_numbers.len(), q);
    let qf = qi.floor();
    let i = qf as usize;

    if (i as f64) == qi {
        return Ok(sorted_numbers[i]);
    }

    let t = qi - qf;

    assert!(sorted_numbers.len() >= (i + 2));

    let x0 = sorted_numbers[i];
    let x1 = sorted_numbers[i + 1];

    return Ok(x0 * (1.0 - t) + x1 * t);
}

fn summarize_numbers(xs: &Vec<f64>, estimators: &Vec<Estimator>) -> Result<(), Error> {
    println!("Count:\t{}", xs.len());

    for est in estimators.iter() {
        let val = (est.func)(xs)?;
        println!("{}:\t{}", est.name, val);
    }

    Ok(())
}

struct Estimator {
    name: String,
    func: fn(&Vec<f64>) -> Result<f64, Error>,
}

#[derive(Debug)]
struct EstimatorResult {
    name: String,
    full_baseline_estimator: f64,
    target_estimator: f64,
    sim_count: i32,
    target_lt_sim_count: i32,
    target_gt_sim_count: i32,
}

fn simulate(
    iterations: i32,
    baseline: &Vec<f64>,
    target: &Vec<f64>,
    estimators: &Vec<Estimator>,
) -> Result<Vec<EstimatorResult>, Error> {
    debug_assert!(is_sorted(baseline));

    let mut results: Vec<(&Estimator, EstimatorResult)> = Vec::new();

    for est in estimators.iter() {
        results.push((
            est,
            EstimatorResult {
                name: est.name.clone(),
                full_baseline_estimator: (est.func)(baseline)?,
                target_estimator: (est.func)(target)?,
                sim_count: 0,
                target_lt_sim_count: 0,
                target_gt_sim_count: 0,
            },
        ));
    }

    let mut rng = rand::thread_rng();

    let mut resampling_vec: Vec<f64> = Vec::new();
    resampling_vec.reserve_exact(target.len());

    for _ in 0..iterations {
        resampling_vec.clear();
        for _ in 0..target.len() {
            let item = rng.gen_range(0..baseline.len());
            resampling_vec.push(baseline[item]);
        }
        resampling_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (est, res) in results.iter_mut() {
            let sim_val = (est.func)(&resampling_vec)?;

            res.sim_count += 1;

            match res
                .target_estimator
                .partial_cmp(&sim_val)
                .expect("estimator should not be NaN")
            {
                Ordering::Less => {
                    res.target_lt_sim_count += 1;
                }
                Ordering::Greater => {
                    res.target_gt_sim_count += 1;
                }
                Ordering::Equal => (),
            }
        }
    }

    Ok(results.into_iter().map(|(_, x)| x).collect())
}

fn main() -> Result<(), Error> {
    let args = Cli::parse();

    let baseline = read_and_sort_numbers(args.baseline_filename)?;
    let target = read_and_sort_numbers(args.target_filename)?;

    let estimators = vec![
        Estimator {
            name: "avg".to_string(),
            func: |xs| Ok(xs.iter().sum::<f64>() / (xs.len() as f64)),
        },
        Estimator {
            name: "min".to_string(),
            func: |xs| get_quantile(xs, 0.0),
        },
        Estimator {
            name: "p50".to_string(),
            func: |xs| get_quantile(xs, 0.5),
        },
        Estimator {
            name: "p75".to_string(),
            func: |xs| get_quantile(xs, 0.75),
        },
        Estimator {
            name: "p90".to_string(),
            func: |xs| get_quantile(xs, 0.9),
        },
        Estimator {
            name: "p95".to_string(),
            func: |xs| get_quantile(xs, 0.95),
        },
        Estimator {
            name: "p99".to_string(),
            func: |xs| get_quantile(xs, 0.99),
        },
        Estimator {
            name: "max".to_string(),
            func: |xs| get_quantile(xs, 1.0),
        },
    ];

    println!("=== Summary (baseline) ===");
    summarize_numbers(&baseline, &estimators)?;
    println!("");

    println!("=== Summary (target) ===");
    summarize_numbers(&target, &estimators)?;
    println!("");

    let results = simulate(args.iterations, &baseline, &target, &estimators)?;
    println!("=== Comparison ===");
    for result in results.iter() {
        if result.target_estimator > result.full_baseline_estimator {
            let r = (result.target_gt_sim_count as f64) / (result.sim_count as f64);
            println!(
                "{}: {} to {}, {}",
                result.name, result.full_baseline_estimator, result.target_estimator, r
            );
        } else {
            let r = (result.target_gt_sim_count as f64) / (result.sim_count as f64);
            println!(
                "{}: {} to {}, {}",
                result.name, result.full_baseline_estimator, result.target_estimator, r
            );
        }
    }

    Ok(())
}
