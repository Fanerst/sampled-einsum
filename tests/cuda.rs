//! The sampled einsum on tenferro's CUDA engine, against a dense CPU einsum followed by
//! host-side indexing: the random networks of `dense_reference.rs` (same seeds and counts),
//! with operands uploaded, every step run on the GPU and the result downloaded.
//!
//! Built only with `--features cuda`. Without a usable CUDA device the tests pass with a
//! message, unless `SAMPLED_EINSUM_REQUIRE_CUDA=1` is set, in which case they fail.
#![cfg(feature = "cuda")]

mod common;

use std::collections::HashMap;

use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::SeedableRng;
use sampled_einsum::{plan, ConcreteOps, StepKind};
use tenferro_cpu::CpuEngine;
use tenferro_gpu::cuda::{cuda_devices, download_tensor, gpu_available, upload_tensor, CudaEngine};
use tenferro_tensor::{Tensor, TensorValue};

use common::{dense, operands_c64, operands_f64, options, random_case, select_dense};

type Result<T> = sampled_einsum::Result<T>;

/// A CUDA engine on the first visible device, or `None` when the host has none.
fn cuda_engine() -> Result<Option<CudaEngine>> {
    let required = std::env::var("SAMPLED_EINSUM_REQUIRE_CUDA").is_ok_and(|v| v == "1");
    let device = if gpu_available() {
        cuda_devices()?.into_iter().next()
    } else {
        None
    };
    match device {
        Some(device) => Ok(Some(CudaEngine::open(device.id())?)),
        None if required => Err("SAMPLED_EINSUM_REQUIRE_CUDA=1, but no usable CUDA device".into()),
        None => {
            eprintln!("no usable CUDA device: skipping");
            Ok(None)
        }
    }
}

fn upload(gpu: &CudaEngine, operands: &[Tensor]) -> Result<Vec<TensorValue>> {
    operands
        .iter()
        .map(|t| Ok(TensorValue::from_tensor(upload_tensor(gpu.runtime(), t)?)))
        .collect()
}

#[test]
fn random_networks_f64_on_cuda_match_dense() -> Result<()> {
    let Some(mut gpu) = cuda_engine()? else {
        return Ok(());
    };
    let mut cpu = CpuEngine::serial();
    let mut rng = StdRng::seed_from_u64(2675);
    let mut seen: HashMap<String, usize> = HashMap::new();
    for _ in 0..400 {
        let case = random_case(&mut rng);
        let operands = operands_f64(&mut rng, &case.net)?;
        let full = dense(&case.net, &operands, &mut cpu)?;
        let (shape, want) = select_dense(&case.net, &case.samples, full.as_slice::<f64>()?);
        let on_gpu = upload(&gpu, &operands)?;
        let refs: Vec<&TensorValue> = on_gpu.iter().collect();
        for (name, opts) in options() {
            let p = plan(&case.net, &case.path, &case.samples, &opts)?;
            let mut ops = ConcreteOps::new(&mut gpu);
            let (y, stats) = p.execute(&mut ops, &refs)?;
            let y = ops.to_tensor(y)?;
            let y = download_tensor(gpu.runtime(), &y)?;
            assert_eq!(y.shape(), shape.as_slice(), "{name}: {:?}", case.net);
            for (k, (g, w)) in y.as_slice::<f64>()?.iter().zip(&want).enumerate() {
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
    eprintln!("step kinds exercised on CUDA: {seen:?}");
    Ok(())
}

#[test]
fn random_networks_c64_on_cuda_match_dense() -> Result<()> {
    let Some(mut gpu) = cuda_engine()? else {
        return Ok(());
    };
    let mut cpu = CpuEngine::serial();
    let mut rng = StdRng::seed_from_u64(1867);
    for _ in 0..150 {
        let case = random_case(&mut rng);
        let operands = operands_c64(&mut rng, &case.net)?;
        let full = dense(&case.net, &operands, &mut cpu)?;
        let (shape, want) = select_dense(&case.net, &case.samples, full.as_slice::<Complex64>()?);
        let on_gpu = upload(&gpu, &operands)?;
        let refs: Vec<&TensorValue> = on_gpu.iter().collect();
        for (name, opts) in options() {
            let p = plan(&case.net, &case.path, &case.samples, &opts)?;
            let mut ops = ConcreteOps::new(&mut gpu);
            let (y, stats) = p.execute(&mut ops, &refs)?;
            let y = ops.to_tensor(y)?;
            let y = download_tensor(gpu.runtime(), &y)?;
            assert_eq!(y.shape(), shape.as_slice(), "{name}");
            for (g, w) in y.as_slice::<Complex64>()?.iter().zip(&want) {
                assert!(
                    (g - w).norm() <= 1e-12 * (1.0 + w.norm()),
                    "{name}: got {g}, want {w}"
                );
            }
            assert_eq!(
                stats.peak_live_elems, p.report.peak_elems,
                "{name}: measured vs planned peak"
            );
        }
    }
    Ok(())
}
