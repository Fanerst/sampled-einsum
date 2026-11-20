//! Random networks, samples and a dense host-side reference shared by the tests.
#![allow(dead_code)]

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use sampled_einsum::{Network, Samples};

pub struct Case {
    pub net: Network,
    pub path: Vec<(usize, usize)>,
    pub samples: Samples,
}

/// A random einsum: 1-6 operands, labels of extent 1-3 (some shared by three or
/// more operands), a random output, a random non-empty sampled subset of it,
/// and a random SSA path.
pub fn random_case(rng: &mut StdRng) -> Case {
    let n_ops = rng.gen_range(1..=6);
    let n_labels = rng.gen_range(2..=7u32);
    let sizes: HashMap<u32, usize> = (0..n_labels).map(|l| (l, rng.gen_range(1..=3))).collect();
    let mut inputs = Vec::new();
    for _ in 0..n_ops {
        let rank = rng.gen_range(1..=4.min(n_labels as usize));
        let mut pool: Vec<u32> = (0..n_labels).collect();
        pool.shuffle(rng);
        inputs.push(pool[..rank].to_vec());
    }
    let mut used: Vec<u32> = inputs.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    used.shuffle(rng);
    let n_out = rng.gen_range(1..=used.len().min(4));
    let output = used[..n_out].to_vec();
    let mut sampled = output.clone();
    sampled.shuffle(rng);
    sampled.truncate(rng.gen_range(1..=output.len()));
    let net = Network {
        inputs,
        output,
        sizes,
    };
    let samples = random_samples(rng, &net, &sampled);
    let path = random_path(rng, n_ops);
    Case { net, path, samples }
}

/// Random samples (with duplicates), or every combination when there are few.
pub fn random_samples(rng: &mut StdRng, net: &Network, labels: &[u32]) -> Samples {
    let dims: Vec<usize> = labels.iter().map(|l| net.sizes[l]).collect();
    let combos: usize = dims.iter().product();
    if combos <= 24 && rng.gen_bool(0.3) {
        // Every combination, in a shuffled order.
        let mut order: Vec<usize> = (0..combos).collect();
        order.shuffle(rng);
        let n = combos;
        let mut values = vec![0; n * labels.len()];
        for (s, &c) in order.iter().enumerate() {
            let mut rem = c;
            for (j, &d) in dims.iter().enumerate() {
                values[s + n * j] = rem % d;
                rem /= d;
            }
        }
        return Samples {
            labels: labels.to_vec(),
            n,
            values,
        };
    }
    let n = rng.gen_range(1..=12);
    let mut values = vec![0; n * labels.len()];
    for (j, &d) in dims.iter().enumerate() {
        for s in 0..n {
            values[s + n * j] = rng.gen_range(0..d);
        }
    }
    Samples {
        labels: labels.to_vec(),
        n,
        values,
    }
}

/// A random SSA contraction path over `n_ops` leaves.
pub fn random_path(rng: &mut StdRng, n_ops: usize) -> Vec<(usize, usize)> {
    let mut live: Vec<usize> = (0..n_ops).collect();
    let mut path = Vec::new();
    for k in 0..n_ops.saturating_sub(1) {
        let i = rng.gen_range(0..live.len());
        let a = live.swap_remove(i);
        let j = rng.gen_range(0..live.len());
        let b = live.swap_remove(j);
        path.push((a, b));
        live.push(n_ops + k);
    }
    path
}

/// `Y[rest..., s]` read from a dense column-major output over `net.output`.
pub fn select_dense<T: Copy>(
    net: &Network,
    samples: &Samples,
    dense: &[T],
) -> (Vec<usize>, Vec<T>) {
    let out_dims: Vec<usize> = net.output.iter().map(|l| net.sizes[l]).collect();
    let column: HashMap<u32, usize> = samples
        .labels
        .iter()
        .enumerate()
        .map(|(j, &l)| (l, j))
        .collect();
    let rest: Vec<usize> = (0..net.output.len())
        .filter(|&k| !column.contains_key(&net.output[k]))
        .collect();
    let rest_dims: Vec<usize> = rest.iter().map(|&k| out_dims[k]).collect();
    let rest_size: usize = rest_dims.iter().product();
    let n = samples.n;
    let mut y = Vec::with_capacity(rest_size * n);
    for s in 0..n {
        for r in 0..rest_size {
            let mut idx = vec![0usize; out_dims.len()];
            let mut rem = r;
            for (&k, &d) in rest.iter().zip(&rest_dims) {
                idx[k] = rem % d;
                rem /= d;
            }
            for (k, l) in net.output.iter().enumerate() {
                if let Some(&j) = column.get(l) {
                    idx[k] = samples.values[s + n * j];
                }
            }
            let mut lin = 0;
            let mut stride = 1;
            for (k, &d) in out_dims.iter().enumerate() {
                lin += idx[k] * stride;
                stride *= d;
            }
            y.push(dense[lin]);
        }
    }
    let mut shape = rest_dims;
    shape.push(n);
    (shape, y)
}
