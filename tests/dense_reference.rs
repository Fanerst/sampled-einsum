//! The sampled einsum against a dense tenferro einsum followed by host-side indexing.

mod common;

use std::collections::HashMap;

use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sampled_einsum::{plan, ConcreteOps, Network, PlanOptions, Samples, StepKind};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumOps, EinsumSubscripts};
use tenferro_tensor::Tensor;

use common::{random_case, select_dense, Case};

type Result<T> = sampled_einsum::Result<T>;

fn options() -> Vec<(&'static str, PlanOptions)> {
    vec![
        ("default", PlanOptions::default()),
        (
            "pairs, tiny chunks",
            PlanOptions {
                max_chunk_elems: 3,
                outer_select_ratio: 0.0,
            },
        ),
        (
            "outer select",
            PlanOptions {
                max_chunk_elems: usize::MAX,
                outer_select_ratio: 1e9,
            },
        ),
    ]
}

fn operands_f64(rng: &mut StdRng, net: &Network) -> Result<Vec<Tensor>> {
    net.inputs
        .iter()
        .map(|ls| {
            let shape: Vec<usize> = ls.iter().map(|l| net.sizes[l]).collect();
            let len = shape.iter().product();
            let data: Vec<f64> = (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect();
            Ok(Tensor::from_vec_col_major(shape, data)?)
        })
        .collect()
}

fn operands_c64(rng: &mut StdRng, net: &Network) -> Result<Vec<Tensor>> {
    net.inputs
        .iter()
        .map(|ls| {
            let shape: Vec<usize> = ls.iter().map(|l| net.sizes[l]).collect();
            let len = shape.iter().product();
            let data: Vec<Complex64> = (0..len)
                .map(|_| Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
                .collect();
            Ok(Tensor::from_vec_col_major(shape, data)?)
        })
        .collect()
}

fn dense(net: &Network, operands: &[Tensor], engine: &mut CpuEngine) -> Result<Tensor> {
    let labels: Vec<&[u32]> = net.inputs.iter().map(Vec::as_slice).collect();
    let refs: Vec<&Tensor> = operands.iter().collect();
    Ok(refs.einsum(&EinsumSubscripts::new(&labels, &net.output), engine)?)
}

/// Run one case with every planner setting; return the step kinds seen.
fn check_f64(case: &Case, rng: &mut StdRng, seen: &mut HashMap<String, usize>) -> Result<()> {
    let mut engine = CpuEngine::serial();
    let operands = operands_f64(rng, &case.net)?;
    let full = dense(&case.net, &operands, &mut engine)?;
    let (shape, want) = select_dense(&case.net, &case.samples, full.as_slice::<f64>()?);
    let refs: Vec<&Tensor> = operands.iter().collect();
    for (name, opts) in options() {
        let p = plan(&case.net, &case.path, &case.samples, &opts)?;
        let (y, stats) = p.execute(&mut ConcreteOps::new(&mut engine), &refs)?;
        assert_eq!(
            y.shape(),
            shape.as_slice(),
            "{name}: {case:?}",
            case = case.net
        );
        let got = y.as_slice::<f64>()?;
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-12 * (1.0 + w.abs()),
                "{name}: entry {k}: got {g}, want {w}\nnet {:?}\npath {:?}\nsamples {:?}",
                case.net,
                case.path,
                case.samples
            );
        }
        assert_eq!(
            stats.peak_live_elems, p.report.peak_elems,
            "{name}: measured vs planned peak"
        );
        for step in &p.report.steps {
            let kind = match step.kind {
                StepKind::Pairs { chunks } if chunks > 1 => "Pairs (chunked)".to_string(),
                ref k => format!("{k:?}"),
            };
            *seen.entry(kind).or_default() += 1;
        }
    }
    Ok(())
}

#[test]
fn random_networks_f64_match_dense() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(2675);
    let mut seen = HashMap::new();
    for _ in 0..400 {
        let case = random_case(&mut rng);
        check_f64(&case, &mut rng, &mut seen)?;
    }
    for kind in [
        "Dense",
        "OneSided",
        "FullProduct",
        "OuterSelect",
        "Pairs { chunks: 1 }",
        "Pairs (chunked)",
    ] {
        assert!(
            seen.get(kind).copied().unwrap_or(0) > 0,
            "step kind {kind} never ran: {seen:?}"
        );
    }
    eprintln!("step kinds exercised: {seen:?}");
    Ok(())
}

#[test]
fn random_networks_c64_match_dense() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(1867);
    let mut engine = CpuEngine::serial();
    for _ in 0..150 {
        let case = random_case(&mut rng);
        let operands = operands_c64(&mut rng, &case.net)?;
        let full = dense(&case.net, &operands, &mut engine)?;
        let (shape, want) = select_dense(&case.net, &case.samples, full.as_slice::<Complex64>()?);
        let refs: Vec<&Tensor> = operands.iter().collect();
        for (name, opts) in options() {
            let p = plan(&case.net, &case.path, &case.samples, &opts)?;
            let (y, _) = p.execute(&mut ConcreteOps::new(&mut engine), &refs)?;
            assert_eq!(y.shape(), shape.as_slice(), "{name}");
            for (g, w) in y.as_slice::<Complex64>()?.iter().zip(&want) {
                assert!(
                    (g - w).norm() <= 1e-12 * (1.0 + w.norm()),
                    "{name}: got {g}, want {w}"
                );
            }
        }
    }
    Ok(())
}

fn run_case(net: &Network, path: &[(usize, usize)], samples: &Samples, seed: u64) -> Result<()> {
    let case = Case {
        net: net.clone(),
        path: path.to_vec(),
        samples: samples.clone(),
    };
    let mut rng = StdRng::seed_from_u64(seed);
    check_f64(&case, &mut rng, &mut HashMap::new())
}

fn sizes(pairs: &[(u32, usize)]) -> HashMap<u32, usize> {
    pairs.iter().copied().collect()
}

#[test]
fn sddmm_and_duplicates() -> Result<()> {
    // ij,jk->ik at (i, k) = (1, 2), (0, 0), (1, 2): a duplicate sample.
    let net = Network {
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        sizes: sizes(&[(0, 3), (1, 4), (2, 5)]),
    };
    let samples = Samples {
        labels: vec![0, 2],
        n: 3,
        values: vec![1, 0, 1, 2, 0, 2],
    };
    run_case(&net, &[(0, 1)], &samples, 1)
}

#[test]
fn one_sample_and_no_samples() -> Result<()> {
    let net = Network {
        inputs: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        output: vec![0, 3],
        sizes: sizes(&[(0, 2), (1, 3), (2, 2), (3, 3)]),
    };
    let one = Samples {
        labels: vec![3],
        n: 1,
        values: vec![2],
    };
    run_case(&net, &[(0, 1), (3, 2)], &one, 2)?;
    let none = Samples {
        labels: vec![3, 0],
        n: 0,
        values: vec![],
    };
    run_case(&net, &[(1, 2), (0, 3)], &none, 3)
}

#[test]
fn sampled_label_on_several_operands() -> Result<()> {
    // b is sampled and sits on all three operands; d is sampled too; a and c stay dense.
    let net = Network {
        inputs: vec![vec![0, 1], vec![1, 2, 4], vec![1, 3, 4]],
        output: vec![0, 1, 2, 3],
        sizes: sizes(&[(0, 2), (1, 3), (2, 2), (3, 3), (4, 2)]),
    };
    let samples = Samples {
        labels: vec![1, 3],
        n: 5,
        values: vec![0, 2, 2, 1, 0, 1, 0, 1, 2, 1],
    };
    run_case(&net, &[(1, 2), (0, 3)], &samples, 4)?;
    run_case(&net, &[(0, 1), (3, 2)], &samples, 5)
}

#[test]
fn single_operand_with_a_summed_label() -> Result<()> {
    // abc->ca sampled on a: c stays dense, b is summed.
    let net = Network {
        inputs: vec![vec![0, 1, 2]],
        output: vec![2, 0],
        sizes: sizes(&[(0, 4), (1, 3), (2, 2)]),
    };
    let samples = Samples {
        labels: vec![0],
        n: 6,
        values: vec![3, 0, 0, 2, 1, 3],
    };
    run_case(&net, &[], &samples, 6)
}

#[test]
fn several_sampled_labels_on_one_leaf() -> Result<()> {
    // The first operand carries both sampled labels, so its leaf merges two axes before the gather.
    let net = Network {
        inputs: vec![vec![0, 1, 2], vec![2, 3]],
        output: vec![3, 1, 0],
        sizes: sizes(&[(0, 3), (1, 2), (2, 4), (3, 2)]),
    };
    let samples = Samples {
        labels: vec![1, 0],
        n: 4,
        values: vec![1, 0, 1, 0, 2, 2, 0, 1],
    };
    run_case(&net, &[(0, 1)], &samples, 7)
}

#[test]
fn invalid_inputs_are_rejected() {
    let net = Network {
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        sizes: sizes(&[(0, 2), (1, 2), (2, 2)]),
    };
    let ok = Samples {
        labels: vec![0],
        n: 1,
        values: vec![1],
    };
    let not_output = Samples {
        labels: vec![1],
        n: 1,
        values: vec![0],
    };
    let out_of_range = Samples {
        labels: vec![2],
        n: 1,
        values: vec![2],
    };
    let none_sampled = Samples {
        labels: vec![],
        n: 1,
        values: vec![],
    };
    let opts = PlanOptions::default();
    assert!(plan(&net, &[(0, 1)], &ok, &opts).is_ok());
    assert!(plan(&net, &[(0, 1)], &not_output, &opts).is_err());
    assert!(plan(&net, &[(0, 1)], &out_of_range, &opts).is_err());
    assert!(plan(&net, &[(0, 1)], &none_sampled, &opts).is_err());
    assert!(plan(&net, &[(0, 0)], &ok, &opts).is_err());
    assert!(plan(&net, &[], &ok, &opts).is_err());
    let repeated = Network {
        inputs: vec![vec![0, 0], vec![0, 2]],
        output: vec![0, 2],
        sizes: sizes(&[(0, 2), (2, 2)]),
    };
    assert!(plan(&repeated, &[(0, 1)], &ok, &opts).is_err());
}
