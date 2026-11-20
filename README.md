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

The end-to-end circuit example needs about 83 GiB of RAM and takes a few minutes on 16 threads:

```bash
cargo run --release --example circuit_amplitudes -- data/n30m14 16
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
let (y, stats) = p.execute(&mut ConcreteOps::new(&mut engine), &[&a, &b])?; // y: [2]
```

The path is in SSA form: step `k` creates node `n_inputs + k`. `EagerOps` runs the same plan on an
`EagerSession`, which records it for reverse mode.

## Layout

| Path | What it holds |
|---|---|
| `src/plan.rs` | The planner, a port of artensor's `contraction_scheme_sparse`, generalized to any extents and to sampled labels shared by several operands. It turns a network, an SSA path and samples into an instruction program and a report. |
| `src/exec.rs` | The `Ops` trait with two implementations: `ConcreteOps` (a `Tensor` on any engine) and `EagerOps` (an `EagerSession`, which records gradients). `SampledPlan::execute` runs a program and measures the peak of live temporaries. |
| `tests/dense_reference.rs` | Random networks and edge cases against a dense tenferro einsum followed by host-side indexing. |
| `tests/ad.rs` | Values and gradients through the eager layer, against the dense route (einsum, then select). |
| `examples/circuit_amplitudes.rs` | Runs the exported circuit network as one sampled einsum and compares the amplitudes with the reference and with artensor. |
| `examples/drop_cost.rs` | Times dropping a 16 GiB tenferro tensor against a `Vec` of the same size. |
| `data/n30m14/` | The 30-qubit, 14-cycle circuit `circuit_n30_m14_s0_e0_pEFGH` as a network: `network.json` (labels, SSA path, 1,000 bitstrings, reference and artensor amplitudes), `tensors.bin` (complex128) and the export log. |
| `scripts/export_circuit.py` | Makes `data/n30m14/` without cirq (see below). |

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
shared server (two Intel Xeon Platinum 8268, 96 threads in all) at a load of about 27.

| | |
|---|---|
| Plan | 4 ms; 223 steps: 160 Dense, 34 OneSided, 23 FullProduct, 6 Pairs (three of them split into 4, 27 and 212 chunks) |
| Accuracy | against the reference amplitudes: fidelity 0.99999999978, max relative difference 1.3e-5 (the file's precision); against artensor (c64): fidelity 0.999999999999, max relative difference 8.8e-7 |
| Memory | planned 48.00 GiB of live temporaries, measured 48.00 GiB; maximum RSS 83 GiB |
| Time | 134 s: einsum 63 s, freeing temporaries 42 s, `index_select` 27 s, concatenate 1.1 s, reshape 0.9 s |

- **Load:** an earlier run on the same server at a load of about 65 took 257 s (einsum 108 s,
  freeing 103 s, `index_select` 43 s), so times there depend on the load.
- **Freeing:** `examples/drop_cost.rs` shows that dropping a 16 GiB tenferro tensor costs the same as
  dropping a `Vec` of that size (1.6–2.6 s against 1.4–2.7 s). That is the OS releasing pages, not
  tenferro overhead. The run spent 356 s of CPU time in the kernel against 487 s in user code, with
  154 million minor page faults (4 KiB pages; the server enables transparent huge pages only on
  request) and few context switches. Fresh large temporaries (gathered copies, chunks and
  intermediates) are faulted in page by page when first written and released when dropped.
- **What v1 removes:** the gathers, the gathered copies and the chunk concatenation. Every step
  still allocates its output, so part of the freeing cost remains.
- **artensor, for orientation only:** 73 s for the same contraction in complex64 with 16 threads on
  another server (two AMD EPYC 9654). That is a different machine and half the precision, so it is
  not a comparison. A fair benchmark row (same machine, same precision, tenferro-benchmark) is
  still to do.

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
- **Backends:** tested on CPU only. `ConcreteOps` takes any `&mut dyn Engine`, so CUDA should work
  but has not been run.
- **Labels:** a label may not repeat within one operand (no diagonals in v0).
- **Traced mode:** not covered. The plan depends on the sample values.

## License

MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`), as tenferro-rs. The planner ports logic from
artensor (Apache-2.0, same author).
