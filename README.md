# sampled-einsum: a tenferro prototype for tensor4all/tenferro-rs#2675

This is v0 of the sampled einsum proposed in
[tensor4all/tenferro-rs#2675](https://github.com/tensor4all/tenferro-rs/issues/2675). Given an
einsum, a contraction path and a list of samples (values of some output labels), it computes

```text
Y[rest..., s] = T[rest..., sample s],   T = einsum(operands)
```

without forming `T`. Each node of the contraction tree keeps one row per distinct partial sample
below it: the sparse-state method of artensor (Pan, Chen and Zhang, PRL 129, 090502 (2022)).
It runs entirely on tenferro's public operations (`einsum`, `index_select`, `reshape`,
`concatenate`), so tenferro's eager layer differentiates through it with no new rules.

## Quick start

The crate needs Rust 1.96 or newer; `rust-toolchain.toml` pins 1.100.0, which rustup installs on
first use. Cargo fetches tenferro-rs at the pinned commit, and `Cargo.lock` is committed.

```bash
git clone https://github.com/Fanerst/sampled-einsum && cd sampled-einsum && cargo test --release
```

The end-to-end circuit example needs about 83 GiB of RAM and takes a few minutes on 16 threads.
Its arguments are the data directory, a thread count or `cuda`, the precision (`c128` or `c64`)
and the number of runs:

```bash
cargo run --release --example circuit_amplitudes -- data/n30m14 16
```

The `cuda` feature runs the same plans on tenferro's CUDA engine: `tests/cuda.rs` and the
example's `cuda` device. At run time tenferro needs an NVIDIA driver, CUDA 12 runtime libraries
and cuTENSOR 2.x (tenferro's
[GPU guide](https://github.com/tensor4all/tenferro-rs/blob/main/docs/guides/devices-and-gpu.md)
lists the versions and the `TENFERRO_*_PATH` overrides). The c128 circuit needs an 80 GB GPU: it
used up to 69 GiB on an A100.

```bash
cargo test --release --features cuda --test cuda
cargo run --release --features cuda --example circuit_amplitudes -- data/n30m14 cuda c128 3
```

A sampled einsum is a plan, built once from the network, the path and the samples, then run on
tenferro operands. For `ij,jk->ik` sampled at `(i, k) = (0, 1)` and `(1, 2)` (an SDDMM):

```rust
let net = Network {
    inputs: vec![vec![0, 1], vec![1, 2]],
    output: vec![0, 2],
    sizes: HashMap::from([(0, 2), (1, 3), (2, 4)]),
};
// Column-major [n, k]: the value of sampled label j in sample s is at s + n * j.
let samples = Samples { labels: vec![0, 2], n: 2, values: vec![0, 1, 1, 2] };
let p = plan(&net, &[(0, 1)], &samples, &PlanOptions::default())?;
// a, b: TensorValue::from_tensor(...) of the two operands, on the engine's device.
let mut ops = ConcreteOps::new(&mut engine);
let (y, stats) = p.execute(&mut ops, &[&a, &b])?;
let y = ops.to_tensor(y)?; // y: [2]
```

The path is in SSA form: step `k` creates node `n_inputs + k`. `EagerOps` runs the same plan on an
`EagerSession`, which records it for reverse mode.

## Layout

| Path | What it holds |
|---|---|
| `src/plan.rs` | The planner, a port of artensor's `contraction_scheme_sparse`, generalized to any extents and to sampled labels shared by several operands. It turns a network, an SSA path and samples into an instruction program and a report. |
| `src/exec.rs` | The `Ops` trait with two implementations: `ConcreteOps` (a `TensorValue` on any engine; gathers and concatenations run on `[prefix, extent, suffix]` views) and `EagerOps` (an `EagerSession`, which records gradients). `SampledPlan::execute` runs a program and measures the peak of live temporaries. |
| `tests/dense_reference.rs` | Random networks and edge cases against a dense tenferro einsum followed by host-side indexing. |
| `tests/ad.rs` | Values and gradients through the eager layer, against the dense route (einsum, then select). |
| `tests/cuda.rs` | With `--features cuda`: the random networks of `dense_reference.rs` (same seeds) run on the CUDA engine, against the dense CPU reference. |
| `examples/circuit_amplitudes.rs` | Runs the exported circuit network as one sampled einsum, on the CPU or on CUDA, in complex128 or complex64, and compares the amplitudes with the reference and with artensor. |
| `examples/drop_cost.rs` | Times dropping a 16 GiB tenferro tensor against a `Vec` of the same size. |
| `examples/einsum_sequence.rs` | Lists the circuit plan's einsum calls with their shapes, and replays them through a first-in-first-out plan cache like the 32-plan cuTENSOR cache of tenferro's CUDA engine. |
| `data/n30m14/` | The 30-qubit, 14-cycle circuit `circuit_n30_m14_s0_e0_pEFGH` as a network: `network.json` (labels, SSA path, 1,000 bitstrings, reference and artensor amplitudes), `tensors.bin` (complex128) and the export log. `einsum_sequence.jsonl` lists the plan's 463 einsum calls with their shapes, as written by `examples/einsum_sequence.rs`. |
| `scripts/export_circuit.py` | Makes `data/n30m14/` without cirq (see below). |
| `scripts/bench_artensor.py` | Times artensor on the exported tree: rebuilds the tree from the path and compiles artensor's own scheme from the exported order. |
| `scripts/bench_pairs.sh` | Runs the example and `bench_artensor.py` in turns (CPU threads or CUDA), each run under `/usr/bin/time -v`. |

## Each step of a plan

Every intermediate is dense over its non-sampled labels, plus one configuration axis. A step
`v = (l, r)` is one of:

| Kind | When | Execution |
|---|---|---|
| `Dense` | neither child carries sampled labels | one einsum |
| `OneSided` | one child does | its configuration axis is a free axis of the einsum |
| `FullProduct` | every pair of child rows is needed | one einsum and a reshape; no gather |
| `OuterSelect` | the full product is at most `outer_select_ratio` × the rows needed | full product, then `index_select` |
| `Pairs { chunks }` | otherwise | per chunk: `index_select` on both children, a batched einsum over the pair axis, then `concatenate` |

Rows of a node are the distinct pairs (left row, right row) over the samples, sorted
right-row-major. With that order, the full product is exactly the column-major merge of the two
child axes.

## Results so far (Rust 1.100.0, release build)

### Tests (`cargo test --release`)

- **Random networks:** 400 f64 and 150 c64, each under three planner settings (default, pairs
  with tiny chunks, outer select). Networks have 1–6 operands, extents 1–3, hyperedges, random SSA
  paths and duplicate samples. Results equal a dense tenferro einsum followed by host-side
  indexing to 1e-12 relative.
- **Coverage:** every step kind ran. The f64 run had 795 FullProduct, 144 Dense, 197 Pairs, 263
  chunked Pairs, 993 OneSided and 524 OuterSelect steps. In every run the measured peak of live
  temporaries equals the planned peak.
- **Edge cases:** SDDMM with duplicate samples; `n = 0` and `n = 1`; a sampled label on three
  operands; one operand with a summed label; two sampled labels on one operand. Invalid inputs
  are rejected.
- **Gradients:** 60 random networks under two settings. Values, and the gradient of `sum(w * y)`
  for every operand through tenferro's eager layer, equal the dense route (einsum, then select)
  to 1e-11. No new derivative rules were needed.

### End-to-end: 30-qubit, 14-cycle circuit, 1,000 bitstrings

`examples/circuit_amplitudes.rs` used artensor's sparse-state tree (10^11.68 multiply-adds,
largest intermediate 2^30 elements), complex128, and tenferro's CPU engine with 16 threads on a
shared server (two Intel Xeon Platinum 8268, 96 threads in all).

| | |
|---|---|
| Plan | 4 ms; 223 steps: 160 Dense, 34 OneSided, 23 FullProduct, 6 Pairs (three of them split into 4, 27 and 212 chunks) |
| Accuracy | against the reference amplitudes: fidelity 0.99999999978, max relative difference 1.3e-5 (the file's precision); against artensor (c64): fidelity 0.999999999999, max relative difference 8.8e-7 |
| Memory | planned 48.00 GiB of live temporaries, measured 48.00 GiB; maximum RSS 83 GiB |
| Time | 108 s: einsum 56 s, freeing temporaries 30 s, `index_select` 20 s, concatenate 1.0 s, reshape 0.8 s (two runs at a load of about 10–14) |

- **Load:** earlier runs on the same server took 134 s at a load of about 27 and 257 s at a
  load of about 65, so times there depend on the load.
- **Freeing:** `examples/drop_cost.rs` shows that dropping a 16 GiB tenferro tensor costs the same as
  dropping a `Vec` of that size (1.6–2.6 s against 1.4–2.7 s). That is the OS releasing pages, not
  tenferro overhead. The run spent 356 s of CPU time in the kernel against 487 s in user code, with
  154 million minor page faults (4 KiB pages; the server enables transparent huge pages only on
  request) and few context switches. Fresh large temporaries (gathered copies, chunks and
  intermediates) are faulted in page by page when first written and released when dropped.
- **What v1 removes:** the gathers, the gathered copies and the chunk concatenation. Every step
  still allocates its output, so part of the freeing cost remains.

### Against artensor on the same machine

`scripts/bench_pairs.sh` runs the prototype and artensor (6504fcd, torch 2.5.1) in turns, each
run a separate process, with the order swapped every round. Both contract the same tree:
`scripts/bench_artensor.py` rebuilds artensor's tree from the exported path and compiles
artensor's own sparse scheme from the exported order, so artensor runs the scheme it ran at
export. Same server as above, 16 threads each, two rounds at a load of about 10–14
(2026-11-22):

| Precision | Prototype | artensor | Ratio |
|---|---|---|---|
| complex128 | 108.1 s, 108.0 s | 72.4 s, 73.6 s | 1.48 |
| complex64 | 69.3 s, 69.2 s | 43.6 s, 44.2 s | 1.58 |

- **Accuracy:** identical. Both give fidelity 0.99999999978 against the reference amplitudes at
  both precisions.
- **Memory:** maximum RSS is 83 GiB (complex128) and 51 GiB (complex64) for the prototype, 48 GiB
  and 24 GiB for artensor. artensor stays at the live-temporary peak; the prototype's extra
  35 GiB and 27 GiB above it are not yet traced.
- **Kernel time:** both pay for page faults: 324 s of system time for the prototype and
  330–343 s for artensor at complex128 (154 and 135 million minor faults).
- **The prototype's time:** its einsums alone take 56 s, three quarters of artensor's whole run;
  the other 52 s are gathers (20 s) and freeing (30 s). v1 removes the gathers and their copies,
  so part of the freeing remains. The merged-axis gathers that helped on the GPU change little
  here: the first version took 112 s, with 22 s of gathers.

### On a GPU

The same pairs with the `cuda` device, on one A100 80GB PCIe of the same server (driver 535 with
the CUDA 12.2 runtime, cuTENSOR 2.6.0; artensor on torch 2.5.1+cu121). Each process uploads the
operands and times three runs; two rounds (2026-11-22):

| Precision | Prototype | artensor | Ratio |
|---|---|---|---|
| complex128 | 8.7–9.3 s | 1.21–1.29 s | 7.3 |
| complex64 | 4.5–5.1 s | 0.85–1.00 s | 5.3 |

- **Tests:** `cargo test --release --features cuda --test cuda` passes. The random networks of
  `dense_reference.rs`, under every planner setting, match the dense CPU reference to 1e-12, and
  the measured peak of live temporaries equals the planned peak.
- **Accuracy:** the same amplitudes as on the CPU and as artensor (fidelity 0.99999999978 against
  the reference amplitudes).
- **Memory:** 48.00 GiB (complex128) and 24.00 GiB (complex64) of live temporaries, as planned and
  as torch's peak allocation for artensor. nvidia-smi showed up to 69 GiB and 35 GiB in use during
  the prototype's runs, against 52 GiB and 26 GiB during artensor's.

#### Where the time goes on the GPU

Profiled with Nsight Systems 2023.2 (CUDA trace; the second run of a process):

| Version | complex128 | complex64 |
|---|---|---|
| Gathers on the 30-axis tensors (the first version) | 17.5 s | 13.2 s |
| Gathers on `[prefix, extent, suffix]` views (this version) | 9.0 s | 4.5 s |
| The same with tenferro's plan cache and pool budget raised (a local patch, below) | 3.8 s | 2.0 s |
| artensor | 1.2 s | 0.85 s |

- **Gathers:** tenferro's CUDA gather has fast kernels up to rank 8 only. The circuit's
  intermediates have about 30 axes of extent 2, so every `index_select` ran the general kernel
  `t4a_cuda_gather_any`, which decodes each element's axes with 64-bit division and searches the
  window axes for each one. The 249 gathers of a run wrote 1.6×10^10 elements in 9.3 s (0.57 ns
  each), the same at both precisions. `ConcreteOps` now reads the operand as a
  `[prefix, extent, suffix]` view, a relabeling of its compact layout, gathers on that and
  relabels the result: 0.89 s. Concatenation does the same, which took it from 0.53 s to 0.02 s.
- **Idle GPU:** after that change the GPU sat idle 5.4 s of each complex128 run while the host
  waited in `cuMemAllocFromPoolAsync`; 22 large allocations took 4.4 s. Two settings in tenferro
  combine. Its cuTENSOR wrapper keeps 32 plans, evicts the oldest first, and drains the stream
  before it destroys an evicted plan, so with more than 32 contraction shapes nearly every
  contraction builds a plan and synchronizes: 130 times per run. At each synchronization the CUDA
  memory pool hands back the idle memory above its retention budget, which tenferro sets to one
  eighth of the device and caps at 4 GiB (`TENFERRO_CUDA_POOL_RELEASE_MIB` is capped the same
  way), and the next multi-GiB allocation maps it again.
- **A local patch:** a build against a tenferro checkout patched to keep 4096 plans and to take a
  budget above 4 GiB (`TENFERRO_CUDA_POOL_RELEASE_MIB=81920`). This was an experiment, not a
  proposed change. First and second run of one process:

  | tenferro | complex128 | complex64 |
  |---|---|---|
  | as pinned | 8.8 s, 9.0 s | 4.8 s, 4.5 s |
  | 4096 plans | 4.4 s, 5.1 s | 3.1 s, 2.6 s |
  | 80 GiB budget | 4.7 s, 3.8 s | 3.1 s, 2.0 s |
  | both | 4.4 s, 3.8 s | 3.0 s, 2.0 s |

- **What remains:** with both, the GPU is busy 99.5% of a run. A complex128 run is then 2.45 s of
  cuTENSOR contraction kernels, 0.89 s of gathers, 0.30 s of memsets and 0.2 s of permutes and
  copies; artensor's run is 0.75 s of cuBLAS GEMMs and 0.43 s of copies. The memsets zero
  503 GiB per run, each right after an allocation and before a gather, contraction or copy.
  The gathers take the same 0.88 s at complex64, so even the rank-3 kernel is limited by its
  integer arithmetic, at about 3 times what its memory traffic needs.

### How `data/n30m14` was made

`scripts/export_circuit.py` reads the circuit and the reference amplitudes from artensor's
`examples/` (artensor 6504fcd, torch 2.9.1, 16 threads):

```bash
python scripts/export_circuit.py ARTENSOR/examples/circuit_n30_m14_s0_e0_pEFGH.py ARTENSOR/examples/amplitudes_n30_m14_s0_e0_pEFGH_10000.txt data/n30m14 1000 30 0
```

The arguments after the output directory are the number of bitstrings, the `sc_target` of the
order search and the seed. The script does not need cirq: it writes out the five gate types the
circuit uses and folds single-qubit gates into the two-qubit ones (224 tensors, 388 labels). It
takes the order from artensor's sparse-state-aware `find_order`, and checks artensor's own
amplitudes against the reference before writing (fidelity 1.00000005, max relative difference
1.3e-5), which validates the gates independently of this crate.

## What v0 leaves out

- **Copies:** gathers and chunk concatenation materialize copies. This is v0 of the issue. The
  gather-free v1 (`t4a-desc::GemmJob` through `cpueinsum::GroupedPlan` on CPU, a pointer-array
  batched GEMM on CUDA) is the next step.
- **Contraction order:** the path is an input. Orders come from artensor's sparse-state-aware
  search, or from any SSA path.
- **GPU speed:** the prototype runs on tenferro's CUDA engine (`ConcreteOps` uploads the gather
  indices there, since the CUDA gather takes only device indices), but 5–7 times slower than
  artensor. Most of the gap is in tenferro (see "Where the time goes on the GPU"): the plan-cache
  synchronizations together with the pool's 4 GiB retention cap, then cuTENSOR's contractions of
  30-mode tensors and the gather kernel's integer arithmetic.
- **Labels:** a label may not repeat within one operand (no diagonals in v0).
- **Traced mode:** not covered. The plan depends on the sample values.

## License

MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`), as tenferro-rs. The planner ports logic from
artensor (Apache-2.0, same author).
