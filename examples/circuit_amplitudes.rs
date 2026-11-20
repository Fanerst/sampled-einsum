//! End-to-end check on the 30-qubit, 14-cycle Sycamore circuit: amplitudes of the
//! sampled bitstrings, computed as one sampled einsum, against the reference
//! amplitudes and artensor's sparse-state result (both from `scripts/export_circuit.py`).
//!
//! Usage: cargo run --release --example circuit_amplitudes -- [DIR] [THREADS]
//! (DIR defaults to `data/n30m14`, THREADS to 1).

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use num_complex::Complex64;
use sampled_einsum::{plan, ConcreteOps, Network, PlanOptions, Samples, StepKind};
use serde::Deserialize;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_tensor::Tensor;

#[derive(Deserialize)]
struct Export {
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    path: Vec<[usize; 2]>,
    bitstrings: Vec<String>,
    reference: Vec<[f64; 2]>,
    artensor: Vec<[f64; 2]>,
    artensor_seconds: serde_json::Value,
    tree_complexity: Vec<f64>,
}

/// `(fidelity, max |a - b| / max |b|)`.
fn compare(a: &[Complex64], b: &[Complex64]) -> (f64, f64) {
    let dot: Complex64 = a.iter().zip(b).map(|(x, y)| y.conj() * x).sum();
    let na: f64 = a.iter().map(|x| x.norm_sqr()).sum();
    let nb: f64 = b.iter().map(|x| x.norm_sqr()).sum();
    let max_b = b.iter().map(|x| x.norm()).fold(0.0, f64::max);
    let max_diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max);
    (dot.norm_sqr() / (na * nb), max_diff / max_b)
}

fn complex(pairs: &[[f64; 2]]) -> Vec<Complex64> {
    pairs.iter().map(|p| Complex64::new(p[0], p[1])).collect()
}

fn main() -> sampled_einsum::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(String::as_str).unwrap_or("data/n30m14");
    let threads: usize = match args.get(2) {
        Some(t) => t.parse()?,
        None => 1,
    };
    let ex: Export =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/network.json"))?)?;
    let bytes = std::fs::read(format!("{dir}/tensors.bin"))?;

    let mut sizes = HashMap::new();
    for &l in ex.inputs.iter().flatten() {
        sizes.insert(l, 2usize);
    }
    let mut offset = 0;
    let mut operands = Vec::new();
    for labels in &ex.inputs {
        let len = 1usize << labels.len();
        let data: Vec<Complex64> = (0..len)
            .map(|k| {
                let at = offset + 16 * k;
                let re = f64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
                let im = f64::from_le_bytes(bytes[at + 8..at + 16].try_into().expect("8 bytes"));
                Complex64::new(re, im)
            })
            .collect();
        offset += 16 * len;
        operands.push(Tensor::from_vec_col_major(vec![2; labels.len()], data)?);
    }
    assert_eq!(offset, bytes.len(), "tensors.bin size");

    let net = Network {
        inputs: ex.inputs.clone(),
        output: ex.output.clone(),
        sizes,
    };
    let n = ex.bitstrings.len();
    let mut values = vec![0usize; n * ex.output.len()];
    for (s, b) in ex.bitstrings.iter().enumerate() {
        for (j, c) in b.chars().enumerate() {
            values[s + n * j] = usize::from(c == '1');
        }
    }
    let samples = Samples {
        labels: ex.output.clone(),
        n,
        values,
    };
    let path: Vec<(usize, usize)> = ex.path.iter().map(|p| (p[0], p[1])).collect();

    let t0 = Instant::now();
    let p = plan(&net, &path, &samples, &PlanOptions::default())?;
    let t_plan = t0.elapsed().as_secs_f64();
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for s in &p.report.steps {
        let k = match s.kind {
            StepKind::Pairs { chunks } => format!("Pairs (max chunks {chunks})"),
            ref k => format!("{k:?}"),
        };
        *kinds.entry(k).or_default() += 1;
    }
    let max_rows = p.report.steps.iter().map(|s| s.rows).max().unwrap_or(0);
    println!(
        "{} operands, {} labels, {} bitstrings; artensor tree: tc 10^{:.2}, sc 2^{:.1}",
        ex.inputs.len(),
        net.sizes.len(),
        n,
        ex.tree_complexity[0],
        ex.tree_complexity[1]
    );
    println!(
        "plan: {:.3} s; {} steps {:?}; max rows {}; multiply-adds 10^{:.2}; planned peak {:.2} GiB (c128)",
        t_plan,
        p.report.steps.len(),
        kinds,
        max_rows,
        p.report.total_madds.log10(),
        p.report.peak_elems as f64 * 16.0 / (1u64 << 30) as f64
    );

    let mut engine = if threads > 1 {
        CpuEngine::new(Pool::builder().threads(threads).build()?)
    } else {
        CpuEngine::serial()
    };
    let refs: Vec<&Tensor> = operands.iter().collect();
    let t0 = Instant::now();
    let (y, stats) = p.execute(&mut ConcreteOps::new(&mut engine), &refs)?;
    let t_run = t0.elapsed().as_secs_f64();
    let amps = y.as_slice::<Complex64>()?;
    println!(
        "tenferro CPU ({threads} threads, c128): {:.2} s; measured peak {:.2} GiB of live temporaries",
        t_run,
        stats.peak_live_elems as f64 * 16.0 / (1u64 << 30) as f64
    );
    let breakdown: Vec<String> = stats
        .seconds
        .iter()
        .map(|(kind, s)| format!("{kind} {s:.2} s"))
        .collect();
    println!("time by operation: {}", breakdown.join(", "));
    let (f_ref, d_ref) = compare(amps, &complex(&ex.reference));
    let (f_art, d_art) = compare(amps, &complex(&ex.artensor));
    println!("vs reference amplitudes: fidelity {f_ref:.12}, max |diff| / max |ref| = {d_ref:.2e}");
    println!("vs artensor (c64):       fidelity {f_art:.12}, max |diff| / max |ref| = {d_art:.2e}");
    println!(
        "artensor timing recorded at export: {}",
        ex.artensor_seconds
    );
    Ok(())
}
