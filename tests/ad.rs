//! Gradients through the eager layer: the sampled plan against a dense einsum
//! followed by a select, both recorded by tenferro's eager tape.

mod common;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sampled_einsum::{plan, EagerOps, Network, PlanOptions, Samples};
use tenferro_ad::{AdContext, EagerRuntime, EagerSession, EagerTensor};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EagerEinsumOps, EinsumSubscripts};
use tenferro_tensor::Tensor;

use common::random_case;

type Result<T> = sampled_einsum::Result<T>;

/// `y` by the dense route: einsum to the full output, move the sampled labels
/// last, merge them, and select one row per sample.
fn dense_route(
    s: &mut EagerSession<'_>,
    net: &Network,
    samples: &Samples,
    vars: &[&EagerTensor],
) -> Result<EagerTensor> {
    let labels: Vec<&[u32]> = net.inputs.iter().map(Vec::as_slice).collect();
    let full = s.einsum(vars, &EinsumSubscripts::new(&labels, &net.output))?;
    let rest: Vec<u32> = net
        .output
        .iter()
        .copied()
        .filter(|l| !samples.labels.contains(l))
        .collect();
    let mut order = rest.clone();
    order.extend(&samples.labels);
    let moved = s.einsum(
        &[&full],
        &EinsumSubscripts::new(&[net.output.as_slice()], &order),
    )?;
    let mut shape: Vec<usize> = rest.iter().map(|l| net.sizes[l]).collect();
    shape.push(samples.labels.iter().map(|l| net.sizes[l]).product());
    let merged = s.reshape(&moved, shape.clone())?;
    let n = samples.n;
    let positions: Vec<usize> = (0..n)
        .map(|i| {
            let mut lin = 0;
            let mut stride = 1;
            for (j, l) in samples.labels.iter().enumerate() {
                lin += samples.values[i + n * j] * stride;
                stride *= net.sizes[l];
            }
            lin
        })
        .collect();
    Ok(s.index_select(&merged, rest.len() as isize, &positions)?)
}

/// `(y, gradient of sum(w * y) for each operand)` along one route.
fn value_and_grads(
    route: &dyn Fn(&mut EagerSession<'_>, &[&EagerTensor]) -> Result<EagerTensor>,
    operands: &[Tensor],
    weights: &Tensor,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    // Einsum's derivative rules live in tenferro-einsum and are installed through the AD context.
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_einsum::semantic_ad_rules()?)?
        .build()?;
    let runtime = EagerRuntime::with_ad_context(&ad)?;
    let mut engine = CpuEngine::serial();
    let mut vars = Vec::new();
    for t in operands {
        vars.push(runtime.variable_from(t.duplicate()?)?);
    }
    let w = runtime.constant_from(weights.duplicate()?)?;
    let mut y_out: Option<Result<Vec<f64>>> = None;
    runtime.with_eager_session(&mut engine, |s| {
        let run = |s: &mut EagerSession<'_>| -> Result<Vec<f64>> {
            let refs: Vec<&EagerTensor> = vars.iter().collect();
            let y = route(s, &refs)?;
            let weighted = s.mul(&y, &w)?;
            let axes: Vec<isize> = (0..weighted.shape().len() as isize).collect();
            let loss = s.reduce_sum(&weighted, Some(&axes))?;
            s.backward(&loss)?;
            Ok(y.value()?.as_slice::<f64>()?.to_vec())
        };
        y_out = Some(run(s));
        Ok::<(), tenferro_ad::Error>(())
    })?;
    let y = y_out.expect("session ran")?;
    let mut grads = Vec::new();
    for v in &vars {
        let g = v.grad()?.ok_or("operand received no gradient")?;
        grads.push(g.as_slice::<f64>()?.to_vec());
    }
    Ok((y, grads))
}

fn assert_close(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-11 * (1.0 + w.abs()),
            "{what}: entry {k}: got {g}, want {w}"
        );
    }
}

#[test]
fn gradients_match_the_dense_route() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(856);
    let settings = [
        PlanOptions::default(),
        PlanOptions {
            max_chunk_elems: 3,
            outer_select_ratio: 0.0,
        },
    ];
    let mut checked = 0;
    while checked < 60 {
        let case = random_case(&mut rng);
        if case.samples.n == 0 {
            continue;
        }
        let operands = case
            .net
            .inputs
            .iter()
            .map(|ls| {
                let shape: Vec<usize> = ls.iter().map(|l| case.net.sizes[l]).collect();
                let len = shape.iter().product();
                Tensor::from_vec_col_major(
                    shape,
                    (0..len)
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect::<Vec<f64>>(),
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let net = case.net.clone();
        let samples = case.samples.clone();
        let reference =
            move |s: &mut EagerSession<'_>, v: &[&EagerTensor]| dense_route(s, &net, &samples, v);
        for opts in &settings {
            let p = plan(&case.net, &case.path, &case.samples, opts)?;
            let shape = p.output_shape().to_vec();
            let len: usize = shape.iter().product();
            let weights = Tensor::from_vec_col_major(
                shape,
                (0..len)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect::<Vec<f64>>(),
            )?;
            let sampled = |s: &mut EagerSession<'_>, v: &[&EagerTensor]| -> Result<EagerTensor> {
                Ok(p.execute(&mut EagerOps::new(s), v)?.0)
            };
            let (y, grads) = value_and_grads(&sampled, &operands, &weights)?;
            let (y_ref, grads_ref) = value_and_grads(&reference, &operands, &weights)?;
            assert_close("y", &y, &y_ref);
            for (i, (g, g_ref)) in grads.iter().zip(&grads_ref).enumerate() {
                assert_close(
                    &format!("grad of operand {i} ({:?})", case.net.inputs[i]),
                    g,
                    g_ref,
                );
            }
        }
        checked += 1;
    }
    Ok(())
}
