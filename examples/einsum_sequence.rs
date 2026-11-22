//! The einsum calls of the circuit plan in program order, and what a first-in-first-out cache of
//! plans keyed by their shapes does with them.
//!
//! tenferro's CUDA engine keeps the cuTENSOR plans of its last 32 shapes and evicts the oldest
//! first. This example lists the plan's einsum calls with their operand and result shapes, counts
//! the distinct ones, and replays the sequence (as repeated runs) through such a cache to count
//! misses and evictions. Only the plan is built; no tensor is allocated.
//!
//! Usage: cargo run --release --example einsum_sequence -- [DIR] [OUT.jsonl]
//!
//! DIR defaults to `data/n30m14`. With OUT.jsonl, every call is written as one JSON line:
//! operand shapes and the result shape, with labels renamed in order of first appearance.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;

use sampled_einsum::{plan, Instr, Network, PlanOptions, Samples};
use serde::{Deserialize, Serialize};

type Result<T> = sampled_einsum::Result<T>;

#[derive(Deserialize)]
struct Export {
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    path: Vec<[usize; 2]>,
    bitstrings: Vec<String>,
}

/// One einsum call: operand shapes then the result shape, and the same for the labels, renamed
/// in order of first appearance so that calls differing only in label names share a key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
struct Call {
    shapes: Vec<Vec<usize>>,
    labels: Vec<Vec<u32>>,
}

/// Replay `calls` `runs` times through a first-in-first-out cache of `capacity` keys (hits do not
/// reorder it). Returns (misses, evictions) per run.
fn fifo(calls: &[Call], capacity: usize, runs: usize) -> Vec<(usize, usize)> {
    let mut order: VecDeque<&Call> = VecDeque::new();
    let mut cached: HashSet<&Call> = HashSet::new();
    (0..runs)
        .map(|_| {
            let (mut misses, mut evictions) = (0, 0);
            for c in calls {
                if cached.contains(c) {
                    continue;
                }
                misses += 1;
                if order.len() >= capacity {
                    let oldest = order.pop_front().expect("a full cache is not empty");
                    cached.remove(oldest);
                    evictions += 1;
                }
                order.push_back(c);
                cached.insert(c);
            }
            (misses, evictions)
        })
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(String::as_str).unwrap_or("data/n30m14");
    let ex: Export =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/network.json"))?)?;
    let sizes: HashMap<u32, usize> = ex.inputs.iter().flatten().map(|&l| (l, 2)).collect();
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
    let p = plan(&net, &path, &samples, &PlanOptions::default())?;

    // Every slot's shape, by running the program on shapes alone.
    let mut shapes: HashMap<usize, Vec<usize>> = ex
        .inputs
        .iter()
        .enumerate()
        .map(|(i, labels)| (i, vec![2; labels.len()]))
        .collect();
    let shape = |shapes: &HashMap<usize, Vec<usize>>, s: usize| -> Result<Vec<usize>> {
        shapes
            .get(&s)
            .cloned()
            .ok_or_else(|| format!("slot {s} read before it was written").into())
    };
    let mut calls = Vec::new();
    for instr in p.program() {
        match instr {
            Instr::Einsum {
                inputs,
                input_labels,
                output_labels,
                out,
            } => {
                let operands = inputs
                    .iter()
                    .map(|&s| shape(&shapes, s))
                    .collect::<Result<Vec<_>>>()?;
                let mut extent = HashMap::new();
                for (labels, dims) in input_labels.iter().zip(&operands) {
                    for (&l, &d) in labels.iter().zip(dims) {
                        extent.insert(l, d);
                    }
                }
                let result: Vec<usize> = output_labels.iter().map(|l| extent[l]).collect();
                let mut rename: HashMap<u32, u32> = HashMap::new();
                let labels = input_labels
                    .iter()
                    .chain(std::iter::once(output_labels))
                    .map(|ls| {
                        ls.iter()
                            .map(|l| {
                                let next = rename.len() as u32;
                                *rename.entry(*l).or_insert(next)
                            })
                            .collect()
                    })
                    .collect();
                let mut call_shapes = operands;
                call_shapes.push(result.clone());
                calls.push(Call {
                    shapes: call_shapes,
                    labels,
                });
                shapes.insert(*out, result);
            }
            Instr::IndexSelect {
                input,
                axis,
                positions,
                out,
            } => {
                let mut s = shape(&shapes, *input)?;
                s[*axis] = positions.len();
                shapes.insert(*out, s);
            }
            Instr::Reshape { shape: s, out, .. } => {
                shapes.insert(*out, s.clone());
            }
            Instr::Concatenate { inputs, axis, out } => {
                let mut s = shape(&shapes, inputs[0])?;
                let mut total = 0;
                for &i in inputs {
                    total += shape(&shapes, i)?[*axis];
                }
                s[*axis] = total;
                shapes.insert(*out, s);
            }
            Instr::Free { slot } => {
                shapes.remove(slot);
            }
        }
    }

    let distinct: HashSet<&Call> = calls.iter().collect();
    let elems = |s: &[usize]| -> usize { s.iter().product() };
    let big = calls
        .iter()
        .filter(|c| elems(c.shapes.last().expect("a result")) >= 1 << 28)
        .count();
    let max_rank = calls
        .iter()
        .flat_map(|c| c.shapes.iter().map(Vec::len))
        .max()
        .unwrap_or(0);
    println!(
        "{} einsum calls per run, {} distinct shapes; operands up to rank {max_rank}; \
         {big} results of at least 2^28 elements (4 GiB at complex128)",
        calls.len(),
        distinct.len()
    );
    // Reuse distance: distinct other shapes seen since the previous call with the same shape,
    // wrapping around into the previous run.
    let mut gaps = Vec::new();
    for (i, c) in calls.iter().enumerate() {
        let mut seen = HashSet::new();
        for k in 1..=calls.len() {
            let d = &calls[(i + calls.len() - k) % calls.len()];
            if d == c {
                break;
            }
            seen.insert(d);
        }
        gaps.push(seen.len());
    }
    gaps.sort_unstable();
    println!(
        "reuse distance in distinct shapes: min {}, median {}, max {}; {} of {} calls within 32",
        gaps[0],
        gaps[gaps.len() / 2],
        gaps[gaps.len() - 1],
        gaps.iter().filter(|&&g| g < 32).count(),
        gaps.len()
    );
    for capacity in [32, 64, 128, 256, 512] {
        let runs = fifo(&calls, capacity, 3);
        println!(
            "FIFO cache of {capacity:3}: misses / evictions per run {:?}",
            runs
        );
    }
    if let Some(out) = args.get(2) {
        let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
        for c in &calls {
            writeln!(f, "{}", serde_json::to_string(c)?)?;
        }
        println!("wrote {} lines to {out}", calls.len());
    }
    Ok(())
}
