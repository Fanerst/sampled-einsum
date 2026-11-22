//! End-to-end check on the 30-qubit, 14-cycle Sycamore circuit: amplitudes of the
//! sampled bitstrings, computed as one sampled einsum, against the reference
//! amplitudes and artensor's sparse-state result (both from `scripts/export_circuit.py`).
//!
//! Usage: cargo run --release [--features cuda] --example circuit_amplitudes --
//!            [DIR] [THREADS | cuda] [c128 | c64] [REPEATS]
//!
//! DIR defaults to `data/n30m14`. THREADS (default 1) runs on a CPU engine with that many
//! threads; `cuda` runs on the first visible CUDA device instead (needs `--features cuda`).
//! The precision defaults to c128 (complex128); c64 is complex64. REPEATS (default 1) runs
//! the same plan that many times on one engine. Every run ends with a JSON summary line.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use num_complex::{Complex32, Complex64};
use sampled_einsum::{plan, ConcreteOps, ExecStats, Network, PlanOptions, Samples, StepKind};
use serde::Deserialize;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_tensor::{Tensor, TensorValue};

type Result<T> = sampled_einsum::Result<T>;

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

/// Host amplitudes as complex128, from a complex128 or complex64 result.
fn amplitudes(y: &Tensor, single: bool) -> Result<Vec<Complex64>> {
    Ok(if single {
        y.as_slice::<Complex32>()?
            .iter()
            .map(|z| Complex64::new(f64::from(z.re), f64::from(z.im)))
            .collect()
    } else {
        y.as_slice::<Complex64>()?.to_vec()
    })
}

/// What every run reports against.
struct Context {
    device: String,
    dtype: &'static str,
    threads: usize,
    elem_bytes: f64,
    reference: Vec<Complex64>,
    artensor: Vec<Complex64>,
}

/// One finished run: amplitudes on the host, the executor's statistics, the seconds from the
/// first operation to the result on the host, and (for CUDA) the upload time before that.
struct Run {
    amps: Vec<Complex64>,
    stats: ExecStats,
    seconds: f64,
    upload_seconds: Option<f64>,
}

fn report(ctx: &Context, k: usize, run: &Run) {
    let gib = |elems: usize| elems as f64 * ctx.elem_bytes / (1u64 << 30) as f64;
    let peak = gib(run.stats.peak_live_elems);
    match run.upload_seconds {
        Some(up) => println!(
            "tenferro {} ({}) run {k}: {:.2} s (operands uploaded in {up:.2} s before); measured peak {peak:.2} GiB of live temporaries",
            ctx.device, ctx.dtype, run.seconds
        ),
        None => println!(
            "tenferro {} ({} threads, {}) run {k}: {:.2} s; measured peak {peak:.2} GiB of live temporaries",
            ctx.device, ctx.threads, ctx.dtype, run.seconds
        ),
    }
    let breakdown: Vec<String> = run
        .stats
        .seconds
        .iter()
        .map(|(kind, s)| format!("{kind} {s:.2} s"))
        .collect();
    let what = if run.upload_seconds.is_some() {
        "host time to submit, by operation"
    } else {
        "time by operation"
    };
    println!("{what}: {}", breakdown.join(", "));
    let (f_ref, d_ref) = compare(&run.amps, &ctx.reference);
    let (f_art, d_art) = compare(&run.amps, &ctx.artensor);
    println!("vs reference amplitudes: fidelity {f_ref:.12}, max |diff| / max |ref| = {d_ref:.2e}");
    println!("vs artensor (c64):       fidelity {f_art:.12}, max |diff| / max |ref| = {d_art:.2e}");
    let summary = serde_json::json!({
        "tool": "tenferro",
        "device": ctx.device,
        "threads": ctx.threads,
        "dtype": ctx.dtype,
        "run": k,
        "seconds": run.seconds,
        "upload_seconds": run.upload_seconds,
        "fidelity": f_ref,
        "max_rel_diff": d_ref,
        "peak_live_gib": peak,
        "by_operation": run.stats.seconds,
    });
    println!("{summary}");
}

fn run_cpu(
    p: &sampled_einsum::SampledPlan,
    operands: &[Tensor],
    ctx: &Context,
    repeats: usize,
) -> Result<()> {
    let mut engine = if ctx.threads > 1 {
        CpuEngine::new(Pool::builder().threads(ctx.threads).build()?)
    } else {
        CpuEngine::serial()
    };
    let values = operands
        .iter()
        .map(|t| Ok(TensorValue::from_tensor(t.duplicate()?)))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&TensorValue> = values.iter().collect();
    for k in 1..=repeats {
        let t0 = Instant::now();
        let mut ops = ConcreteOps::new(&mut engine);
        let (y, stats) = p.execute(&mut ops, &refs)?;
        let y = ops.to_tensor(y)?;
        let seconds = t0.elapsed().as_secs_f64();
        let amps = amplitudes(&y, ctx.dtype == "c64")?;
        let run = Run {
            amps,
            stats,
            seconds,
            upload_seconds: None,
        };
        report(ctx, k, &run);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_cuda(
    p: &sampled_einsum::SampledPlan,
    operands: &[Tensor],
    ctx: &Context,
    repeats: usize,
) -> Result<()> {
    use tenferro_gpu::cuda::{
        cuda_devices, download_tensor, gpu_available, upload_tensor, CudaEngine,
    };
    if !gpu_available() {
        return Err("no usable CUDA driver or device".into());
    }
    let device = cuda_devices()?
        .into_iter()
        .next()
        .ok_or("no CUDA device is visible")?;
    let mut engine = CudaEngine::open(device.id())?;
    for k in 1..=repeats {
        let t0 = Instant::now();
        let on_gpu = operands
            .iter()
            .map(|t| {
                Ok(TensorValue::from_tensor(upload_tensor(
                    engine.runtime(),
                    t,
                )?))
            })
            .collect::<Result<Vec<_>>>()?;
        engine.runtime().synchronize()?;
        let upload_seconds = t0.elapsed().as_secs_f64();
        let refs: Vec<&TensorValue> = on_gpu.iter().collect();
        let t0 = Instant::now();
        let mut ops = ConcreteOps::new(&mut engine);
        let (y, stats) = p.execute(&mut ops, &refs)?;
        let y = ops.to_tensor(y)?;
        // The download waits for every queued operation, so this is the end-to-end time.
        let host = download_tensor(engine.runtime(), &y)?;
        let seconds = t0.elapsed().as_secs_f64();
        drop(y);
        drop(on_gpu);
        let amps = amplitudes(&host, ctx.dtype == "c64")?;
        let run = Run {
            amps,
            stats,
            seconds,
            upload_seconds: Some(upload_seconds),
        };
        report(ctx, k, &run);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(String::as_str).unwrap_or("data/n30m14");
    let device = args.get(2).map(String::as_str).unwrap_or("1");
    let (on_cuda, threads) = if device == "cuda" {
        if !cfg!(feature = "cuda") {
            return Err("this binary was built without the `cuda` feature".into());
        }
        (true, 0)
    } else {
        (false, device.parse::<usize>()?)
    };
    let dtype: &'static str = match args.get(3).map(String::as_str).unwrap_or("c128") {
        "c128" => "c128",
        "c64" => "c64",
        other => return Err(format!("precision must be c128 or c64, not {other}").into()),
    };
    let single = dtype == "c64";
    let repeats: usize = match args.get(4) {
        Some(r) => r.parse()?,
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
        let shape = vec![2; labels.len()];
        operands.push(if single {
            let data: Vec<Complex32> = data
                .iter()
                .map(|z| Complex32::new(z.re as f32, z.im as f32))
                .collect();
            Tensor::from_vec_col_major(shape, data)?
        } else {
            Tensor::from_vec_col_major(shape, data)?
        });
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
    let elem_bytes = if single { 8.0 } else { 16.0 };
    println!(
        "{} operands, {} labels, {} bitstrings; artensor tree: tc 10^{:.2}, sc 2^{:.1}",
        ex.inputs.len(),
        net.sizes.len(),
        n,
        ex.tree_complexity[0],
        ex.tree_complexity[1]
    );
    println!(
        "plan: {:.3} s; {} steps {:?}; max rows {}; multiply-adds 10^{:.2}; planned peak {:.2} GiB ({dtype})",
        t_plan,
        p.report.steps.len(),
        kinds,
        max_rows,
        p.report.total_madds.log10(),
        p.report.peak_elems as f64 * elem_bytes / (1u64 << 30) as f64
    );
    println!(
        "artensor timing recorded at export: {}",
        ex.artensor_seconds
    );

    let ctx = Context {
        device: if on_cuda { "CUDA".into() } else { "CPU".into() },
        dtype,
        threads,
        elem_bytes,
        reference: complex(&ex.reference),
        artensor: complex(&ex.artensor),
    };
    if on_cuda {
        #[cfg(feature = "cuda")]
        run_cuda(&p, &operands, &ctx, repeats)?;
    } else {
        run_cpu(&p, &operands, &ctx, repeats)?;
    }
    Ok(())
}
