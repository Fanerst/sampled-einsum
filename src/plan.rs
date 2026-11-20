//! Planning of a sampled einsum along a fixed contraction path.
//!
//! This is a port of artensor's `contraction_scheme_sparse`, generalized to any
//! label extents and to sampled labels that appear on several operands.
//!
//! Every node of the contraction tree is stored as a dense tensor over its
//! non-sampled labels plus one *configuration axis*. Row `p` of that axis is one
//! distinct assignment `P_v[p]` of the sampled labels below the node. A sample's
//! assignment restricted to `O_l ∪ O_r` is fixed by its rows in the two children,
//! so the rows of a node are the distinct pairs `(row in left, row in right)` over
//! the samples, sorted right-row-major. With that order the full product
//! `P_l × P_r` is exactly the column-major merge of the two child axes, which is
//! what the no-gather step relies on.

use std::collections::{HashMap, HashSet};

use crate::{err, Result};

/// An einsum written with integer labels.
#[derive(Clone, Debug)]
pub struct Network {
    /// Labels of each operand, in axis order. A label may not repeat within one operand.
    pub inputs: Vec<Vec<u32>>,
    /// Labels of the dense output, in axis order.
    pub output: Vec<u32>,
    /// Extent of every label.
    pub sizes: HashMap<u32, usize>,
}

/// The output configurations to evaluate.
#[derive(Clone, Debug)]
pub struct Samples {
    /// Output labels fixed by each sample: a non-empty subset of [`Network::output`].
    pub labels: Vec<u32>,
    /// Number of samples. Duplicates are allowed.
    pub n: usize,
    /// Column-major `[n, labels.len()]`: entry `(s, j)` at `s + n * j` is the
    /// value of `labels[j]` in sample `s`.
    pub values: Vec<usize>,
}

/// Planner settings.
#[derive(Clone, Debug)]
pub struct PlanOptions {
    /// Largest element count of one chunk (the per-chunk einsum output and the
    /// gathered operand copies) in a paired step. The step's result itself is
    /// not split: it is the next node.
    pub max_chunk_elems: usize,
    /// When both children carry configurations, take the full product
    /// `P_l × P_r` and then select the needed rows if that product is at most
    /// this many times the rows needed. `0.0` disables it.
    pub outer_select_ratio: f64,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            max_chunk_elems: 1 << 26,
            outer_select_ratio: 2.0,
        }
    }
}

/// One primitive operation of a plan. Slots `0..n_inputs` are the operands;
/// every other slot is a temporary owned by the executor.
#[derive(Clone, Debug)]
pub enum Instr {
    /// `out = einsum(inputs)` with integer labels.
    Einsum {
        inputs: Vec<usize>,
        input_labels: Vec<Vec<u32>>,
        output_labels: Vec<u32>,
        out: usize,
    },
    /// `out = index_select(input, axis, positions)`.
    IndexSelect {
        input: usize,
        axis: usize,
        positions: Vec<usize>,
        out: usize,
    },
    /// `out = reshape(input, shape)` (column-major).
    Reshape {
        input: usize,
        shape: Vec<usize>,
        out: usize,
    },
    /// `out = concatenate(inputs, axis)`.
    Concatenate {
        inputs: Vec<usize>,
        axis: usize,
        out: usize,
    },
    /// Drop a temporary that is no longer needed.
    Free { slot: usize },
}

/// How one pairwise step was executed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepKind {
    /// Neither child carries sampled labels: one ordinary einsum.
    Dense,
    /// One child carries sampled labels: its configuration axis is a free axis.
    OneSided,
    /// Every pair of child rows is needed: one einsum and a reshape, no gather.
    FullProduct,
    /// The full product is computed and the needed rows are selected.
    OuterSelect,
    /// Rows are gathered per pair and contracted as a batch, in `chunks` chunks.
    Pairs { chunks: usize },
}

/// Planning record of one pairwise step.
#[derive(Clone, Debug)]
pub struct StepReport {
    /// Execution strategy of the step.
    pub kind: StepKind,
    /// Configuration rows of the left child (1 if it has none).
    pub rows_left: usize,
    /// Configuration rows of the right child (1 if it has none).
    pub rows_right: usize,
    /// Configuration rows of the result (1 if it has none).
    pub rows: usize,
    /// Elements of the result.
    pub out_elems: usize,
    /// Multiply-adds of the step's einsum(s).
    pub madds: f64,
}

/// What a plan does, in numbers.
#[derive(Clone, Debug, Default)]
pub struct PlanReport {
    /// Configuration rows of each leaf (1 for a leaf without sampled labels).
    pub leaf_rows: Vec<usize>,
    /// One record per pairwise step, in execution order.
    pub steps: Vec<StepReport>,
    /// Largest total element count of live temporaries (operands excluded).
    pub peak_elems: usize,
    /// Sum of the steps' multiply-adds.
    pub total_madds: f64,
}

/// A planned sampled einsum: run it with [`SampledPlan::execute`](crate::SampledPlan::execute).
#[derive(Clone, Debug)]
pub struct SampledPlan {
    pub(crate) n_inputs: usize,
    pub(crate) input_shapes: Vec<Vec<usize>>,
    pub(crate) n_slots: usize,
    pub(crate) program: Vec<Instr>,
    pub(crate) result: usize,
    pub(crate) output_shape: Vec<usize>,
    /// Planning numbers.
    pub report: PlanReport,
}

impl SampledPlan {
    /// The instruction program, for inspection.
    pub fn program(&self) -> &[Instr] {
        &self.program
    }

    /// Shape of the result: the non-sampled output labels in output order, then the samples.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
}

#[derive(Clone, Debug)]
struct Cfg {
    label: u32,
    rows: usize,
    sample_rows: Vec<u32>,
}

#[derive(Clone, Debug)]
struct Node {
    slot: usize,
    labels: Vec<u32>,
    cfg: Option<Cfg>,
}

struct Builder {
    program: Vec<Instr>,
    slot_elems: Vec<usize>,
    dims: HashMap<u32, usize>,
    next_label: u32,
    n_inputs: usize,
}

impl Builder {
    fn new_slot(&mut self, elems: usize) -> usize {
        self.slot_elems.push(elems);
        self.slot_elems.len() - 1
    }

    fn fresh_label(&mut self, extent: usize) -> u32 {
        let label = self.next_label;
        self.next_label += 1;
        self.dims.insert(label, extent);
        label
    }

    fn elems(&self, labels: &[u32]) -> usize {
        labels.iter().map(|l| self.dims[l]).product()
    }

    fn free_if_temp(&mut self, slot: usize) {
        if slot >= self.n_inputs {
            self.program.push(Instr::Free { slot });
        }
    }

    fn einsum(
        &mut self,
        inputs: Vec<usize>,
        input_labels: Vec<Vec<u32>>,
        output_labels: Vec<u32>,
    ) -> usize {
        let out = self.new_slot(self.elems(&output_labels));
        self.program.push(Instr::Einsum {
            inputs,
            input_labels,
            output_labels,
            out,
        });
        out
    }

    fn index_select(
        &mut self,
        input: usize,
        axis: usize,
        positions: Vec<usize>,
        out_elems: usize,
    ) -> usize {
        let out = self.new_slot(out_elems);
        self.program.push(Instr::IndexSelect {
            input,
            axis,
            positions,
            out,
        });
        out
    }

    fn reshape(&mut self, input: usize, shape: Vec<usize>) -> usize {
        let out = self.new_slot(shape.iter().product());
        self.program.push(Instr::Reshape { input, shape, out });
        out
    }

    /// Multiply-adds of a binary contraction: the product of the extents of every label involved.
    fn madds(&self, lhs: &[u32], rhs: &[u32]) -> f64 {
        let mut seen = HashSet::new();
        lhs.iter()
            .chain(rhs)
            .filter(|l| seen.insert(**l))
            .map(|l| self.dims[l] as f64)
            .product()
    }
}

/// Sorted distinct keys, and each key's rank among them.
fn rank_unique(keys: &[u64]) -> (Vec<u64>, Vec<u32>) {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_unstable_by_key(|&i| keys[i]);
    let mut uniq: Vec<u64> = Vec::new();
    let mut rank = vec![0u32; keys.len()];
    for i in order {
        if uniq.last() != Some(&keys[i]) {
            uniq.push(keys[i]);
        }
        rank[i] = (uniq.len() - 1) as u32;
    }
    (uniq, rank)
}

/// Output labels in tenferro's canonical binary order: left-free, right-free, then batch.
fn canonical_out(lhs: &[u32], rhs: &[u32], keep: impl Fn(u32) -> bool) -> Vec<u32> {
    let in_lhs: HashSet<u32> = lhs.iter().copied().collect();
    let in_rhs: HashSet<u32> = rhs.iter().copied().collect();
    let mut out = Vec::new();
    out.extend(
        lhs.iter()
            .copied()
            .filter(|&x| keep(x) && !in_rhs.contains(&x)),
    );
    out.extend(
        rhs.iter()
            .copied()
            .filter(|&x| keep(x) && !in_lhs.contains(&x)),
    );
    out.extend(
        lhs.iter()
            .copied()
            .filter(|&x| keep(x) && in_rhs.contains(&x)),
    );
    out
}

fn validate(net: &Network, path: &[(usize, usize)], samples: &Samples) -> Result<()> {
    if net.inputs.is_empty() {
        return Err(err("a network needs at least one operand"));
    }
    for (i, labels) in net.inputs.iter().enumerate() {
        let mut seen = HashSet::new();
        for l in labels {
            if !seen.insert(l) {
                return Err(err(format!(
                    "label {l} repeats within operand {i}; v0 does not take diagonals"
                )));
            }
            match net.sizes.get(l) {
                Some(&d) if d > 0 => {}
                _ => return Err(err(format!("label {l} has no positive extent"))),
            }
        }
    }
    let mut seen = HashSet::new();
    for l in &net.output {
        if !seen.insert(l) {
            return Err(err(format!("output label {l} repeats")));
        }
        if !net.inputs.iter().any(|ls| ls.contains(l)) {
            return Err(err(format!("output label {l} is on no operand")));
        }
    }
    if samples.labels.is_empty() {
        return Err(err("at least one output label must be sampled"));
    }
    let mut seen = HashSet::new();
    for l in &samples.labels {
        if !seen.insert(l) {
            return Err(err(format!("sampled label {l} repeats")));
        }
        if !net.output.contains(l) {
            return Err(err(format!("sampled label {l} is not an output label")));
        }
    }
    if samples.values.len() != samples.n * samples.labels.len() {
        return Err(err(format!(
            "values hold {} entries, expected n * labels = {}",
            samples.values.len(),
            samples.n * samples.labels.len()
        )));
    }
    if samples.n > u32::MAX as usize {
        return Err(err("more than 2^32 - 1 samples"));
    }
    for (j, l) in samples.labels.iter().enumerate() {
        let d = net.sizes[l];
        if let Some(v) = samples.values[j * samples.n..(j + 1) * samples.n]
            .iter()
            .find(|&&v| v >= d)
        {
            return Err(err(format!(
                "sample value {v} is out of range for label {l} of extent {d}"
            )));
        }
    }
    let n_in = net.inputs.len();
    if path.len() + 1 != n_in {
        return Err(err(format!(
            "a path over {n_in} operands needs {} steps, got {}",
            n_in - 1,
            path.len()
        )));
    }
    let mut live: HashSet<usize> = (0..n_in).collect();
    for (k, &(a, b)) in path.iter().enumerate() {
        if a == b || !live.remove(&a) || !live.remove(&b) {
            return Err(err(format!(
                "path step {k} ({a}, {b}) does not take two live nodes"
            )));
        }
        live.insert(n_in + k);
    }
    Ok(())
}

/// Plan a sampled einsum along an SSA contraction path.
///
/// `path[k] = (a, b)` contracts nodes `a` and `b` into node `n_inputs + k`
/// (leaves are `0..n_inputs`). The result has shape `[rest..., n]`: the
/// non-sampled output labels in output order, then one axis over the samples.
pub fn plan(
    net: &Network,
    path: &[(usize, usize)],
    samples: &Samples,
    opts: &PlanOptions,
) -> Result<SampledPlan> {
    validate(net, path, samples)?;
    let n_inputs = net.inputs.len();
    let n = samples.n;
    // With no samples, plan one dummy sample and select none of it at the end.
    let (n_eff, values): (usize, Vec<usize>) = if n == 0 {
        (1, vec![0; samples.labels.len()])
    } else {
        (n, samples.values.clone())
    };
    let column: HashMap<u32, usize> = samples
        .labels
        .iter()
        .enumerate()
        .map(|(j, &l)| (l, j))
        .collect();
    let rest: Vec<u32> = net
        .output
        .iter()
        .copied()
        .filter(|l| !column.contains_key(l))
        .collect();
    let rest_set: HashSet<u32> = rest.iter().copied().collect();
    let max_label = net
        .inputs
        .iter()
        .flatten()
        .chain(&net.output)
        .chain(net.sizes.keys())
        .copied()
        .max()
        .unwrap_or(0);
    let mut b = Builder {
        program: Vec::new(),
        slot_elems: Vec::new(),
        dims: net.sizes.clone(),
        next_label: max_label + 1,
        n_inputs,
    };
    let mut report = PlanReport::default();
    let input_shapes: Vec<Vec<usize>> = net
        .inputs
        .iter()
        .map(|ls| ls.iter().map(|l| net.sizes[l]).collect())
        .collect();
    for labels in &net.inputs {
        let elems = b.elems(labels);
        b.new_slot(elems);
    }

    // How many live nodes carry each dense (non-sampled) label.
    let mut count: HashMap<u32, usize> = HashMap::new();
    let mut nodes: Vec<Option<Node>> = Vec::new();
    for (i, labels) in net.inputs.iter().enumerate() {
        for l in labels.iter().filter(|l| !column.contains_key(l)) {
            *count.entry(*l).or_default() += 1;
        }
        let q: Vec<u32> = labels
            .iter()
            .copied()
            .filter(|l| column.contains_key(l))
            .collect();
        if q.is_empty() {
            report.leaf_rows.push(1);
            nodes.push(Some(Node {
                slot: i,
                labels: labels.clone(),
                cfg: None,
            }));
            continue;
        }
        // Linear index of each sample over this operand's sampled labels, first label fastest.
        let mut lin = vec![0u64; n_eff];
        let mut stride = 1u64;
        for l in &q {
            let col = column[l];
            for (s, v) in lin.iter_mut().enumerate() {
                *v += stride * values[s + n_eff * col] as u64;
            }
            stride *= net.sizes[l] as u64;
        }
        let (positions, sample_rows) = rank_unique(&lin);
        let rows = positions.len();
        report.leaf_rows.push(rows);
        let positions: Vec<usize> = positions.into_iter().map(|p| p as usize).collect();
        let (source, mut node_labels, axis) = if q.len() == 1 {
            let axis = labels
                .iter()
                .position(|&x| x == q[0])
                .expect("sampled label is on this operand");
            (i, labels.clone(), axis)
        } else {
            // Move the sampled labels to the end (in `q` order) and merge them into one axis.
            let mut permuted: Vec<u32> =
                labels.iter().copied().filter(|l| !q.contains(l)).collect();
            let kept = permuted.len();
            permuted.extend(&q);
            let t1 = b.einsum(vec![i], vec![labels.clone()], permuted.clone());
            let mut shape: Vec<usize> = permuted[..kept].iter().map(|l| b.dims[l]).collect();
            shape.push(stride as usize);
            let t2 = b.reshape(t1, shape);
            b.free_if_temp(t1);
            let mut merged_labels = permuted[..kept].to_vec();
            merged_labels.push(u32::MAX); // replaced by the configuration label below
            (t2, merged_labels, kept)
        };
        let c = b.fresh_label(rows);
        node_labels[axis] = c;
        let out_elems = b.elems(&node_labels);
        let slot = b.index_select(source, axis, positions, out_elems);
        if source != i {
            b.free_if_temp(source);
        }
        nodes.push(Some(Node {
            slot,
            labels: node_labels,
            cfg: Some(Cfg {
                label: c,
                rows,
                sample_rows,
            }),
        }));
    }

    for &(ia, ib) in path {
        let l = nodes[ia].take().expect("validated path");
        let r = nodes[ib].take().expect("validated path");
        let cfg_labels: HashSet<u32> = l.cfg.iter().chain(&r.cfg).map(|c| c.label).collect();
        for x in l
            .labels
            .iter()
            .chain(&r.labels)
            .filter(|x| !cfg_labels.contains(x))
        {
            *count.get_mut(x).expect("counted") -= 1;
        }
        let survive = |x: u32| rest_set.contains(&x) || count.get(&x).copied().unwrap_or(0) > 0;
        let rows_left = l.cfg.as_ref().map_or(1, |c| c.rows);
        let rows_right = r.cfg.as_ref().map_or(1, |c| c.rows);
        let (node, kind, madds) = match (&l.cfg, &r.cfg) {
            (None, None) => {
                let out = canonical_out(&l.labels, &r.labels, survive);
                let madds = b.madds(&l.labels, &r.labels);
                let slot = b.einsum(
                    vec![l.slot, r.slot],
                    vec![l.labels.clone(), r.labels.clone()],
                    out.clone(),
                );
                (
                    Node {
                        slot,
                        labels: out,
                        cfg: None,
                    },
                    StepKind::Dense,
                    madds,
                )
            }
            (Some(c), None) | (None, Some(c)) => {
                let out = canonical_out(&l.labels, &r.labels, |x| x == c.label || survive(x));
                let madds = b.madds(&l.labels, &r.labels);
                let slot = b.einsum(
                    vec![l.slot, r.slot],
                    vec![l.labels.clone(), r.labels.clone()],
                    out.clone(),
                );
                (
                    Node {
                        slot,
                        labels: out,
                        cfg: Some(c.clone()),
                    },
                    StepKind::OneSided,
                    madds,
                )
            }
            (Some(cl), Some(cr)) => {
                let keys: Vec<u64> = (0..n_eff)
                    .map(|s| ((cr.sample_rows[s] as u64) << 32) | cl.sample_rows[s] as u64)
                    .collect();
                let (pairs, sample_rows) = rank_unique(&keys);
                let rows = pairs.len();
                let product = cl.rows * cr.rows;
                let dense = canonical_out(&l.labels, &r.labels, |x| {
                    !cfg_labels.contains(&x) && survive(x)
                });
                let dense_elems = b.elems(&dense);
                let outer = rows == product
                    || (opts.outer_select_ratio > 0.0
                        && product as f64 <= opts.outer_select_ratio * rows as f64
                        && product.saturating_mul(dense_elems) <= opts.max_chunk_elems);
                if outer {
                    let mut out = dense.clone();
                    out.push(cl.label);
                    out.push(cr.label);
                    let madds = b.madds(&l.labels, &r.labels);
                    let t = b.einsum(
                        vec![l.slot, r.slot],
                        vec![l.labels.clone(), r.labels.clone()],
                        out,
                    );
                    let mut shape: Vec<usize> = dense.iter().map(|x| b.dims[x]).collect();
                    shape.push(product);
                    let merged = b.reshape(t, shape);
                    b.free_if_temp(t);
                    let c = b.fresh_label(rows);
                    let mut labels = dense.clone();
                    labels.push(c);
                    let (slot, kind) = if rows == product {
                        (merged, StepKind::FullProduct)
                    } else {
                        // Row (a, b) of the product sits at a + rows_left * b.
                        let positions: Vec<usize> = pairs
                            .iter()
                            .map(|&k| (k & 0xffff_ffff) as usize + cl.rows * (k >> 32) as usize)
                            .collect();
                        let out_elems = b.elems(&labels);
                        let slot = b.index_select(merged, dense.len(), positions, out_elems);
                        b.free_if_temp(merged);
                        (slot, StepKind::OuterSelect)
                    };
                    (
                        Node {
                            slot,
                            labels,
                            cfg: Some(Cfg {
                                label: c,
                                rows,
                                sample_rows,
                            }),
                        },
                        kind,
                        madds,
                    )
                } else {
                    let p = b.fresh_label(rows);
                    let lhs: Vec<u32> = l
                        .labels
                        .iter()
                        .map(|&x| if x == cl.label { p } else { x })
                        .collect();
                    let rhs: Vec<u32> = r
                        .labels
                        .iter()
                        .map(|&x| if x == cr.label { p } else { x })
                        .collect();
                    let out = canonical_out(&lhs, &rhs, |x| x == p || survive(x));
                    let axis_l = l
                        .labels
                        .iter()
                        .position(|&x| x == cl.label)
                        .expect("cfg axis");
                    let axis_r = r
                        .labels
                        .iter()
                        .position(|&x| x == cr.label)
                        .expect("cfg axis");
                    let axis_out = out
                        .iter()
                        .position(|&x| x == p)
                        .expect("pair axis survives");
                    let per_row = (b.elems(&out) / rows)
                        .max(b.elems(&l.labels) / cl.rows)
                        .max(b.elems(&r.labels) / cr.rows)
                        .max(1);
                    let chunk = (opts.max_chunk_elems / per_row).max(1);
                    let left_rows: Vec<usize> =
                        pairs.iter().map(|&k| (k & 0xffff_ffff) as usize).collect();
                    let right_rows: Vec<usize> =
                        pairs.iter().map(|&k| (k >> 32) as usize).collect();
                    let madds = b.madds(&lhs, &rhs);
                    let mut pieces = Vec::new();
                    let mut start = 0;
                    while start < rows {
                        let end = (start + chunk).min(rows);
                        let m = end - start;
                        let ga = b.index_select(
                            l.slot,
                            axis_l,
                            left_rows[start..end].to_vec(),
                            b.elems(&l.labels) / cl.rows * m,
                        );
                        let gb = b.index_select(
                            r.slot,
                            axis_r,
                            right_rows[start..end].to_vec(),
                            b.elems(&r.labels) / cr.rows * m,
                        );
                        let piece_elems = b.elems(&out) / rows * m;
                        let piece = b.new_slot(piece_elems);
                        b.program.push(Instr::Einsum {
                            inputs: vec![ga, gb],
                            input_labels: vec![lhs.clone(), rhs.clone()],
                            output_labels: out.clone(),
                            out: piece,
                        });
                        b.free_if_temp(ga);
                        b.free_if_temp(gb);
                        pieces.push(piece);
                        start = end;
                    }
                    let chunks = pieces.len();
                    let slot = if chunks == 1 {
                        pieces[0]
                    } else {
                        let slot = b.new_slot(b.elems(&out));
                        b.program.push(Instr::Concatenate {
                            inputs: pieces.clone(),
                            axis: axis_out,
                            out: slot,
                        });
                        for piece in pieces {
                            b.free_if_temp(piece);
                        }
                        slot
                    };
                    (
                        Node {
                            slot,
                            labels: out,
                            cfg: Some(Cfg {
                                label: p,
                                rows,
                                sample_rows,
                            }),
                        },
                        StepKind::Pairs { chunks },
                        madds,
                    )
                }
            }
        };
        b.free_if_temp(l.slot);
        b.free_if_temp(r.slot);
        for x in node
            .labels
            .iter()
            .filter(|x| Some(**x) != node.cfg.as_ref().map(|c| c.label))
        {
            *count.entry(*x).or_default() += 1;
        }
        let rows = node.cfg.as_ref().map_or(1, |c| c.rows);
        report.steps.push(StepReport {
            kind,
            rows_left,
            rows_right,
            rows,
            out_elems: b.elems(&node.labels),
            madds,
        });
        report.total_madds += madds;
        nodes.push(Some(node));
    }

    let root = nodes
        .iter_mut()
        .rev()
        .find_map(Option::take)
        .expect("one root");
    let cfg = root
        .cfg
        .as_ref()
        .expect("every sampled label is on some operand");
    // Bring the root into [rest..., configurations]; this also sums any label that is not an output.
    let mut target = rest.clone();
    target.push(cfg.label);
    let mut slot = root.slot;
    if root.labels != target {
        slot = b.einsum(vec![root.slot], vec![root.labels.clone()], target.clone());
        b.free_if_temp(root.slot);
    }
    // Put the rows in sample order (duplicates allowed).
    let positions: Vec<usize> = if n == 0 {
        Vec::new()
    } else {
        cfg.sample_rows.iter().map(|&r| r as usize).collect()
    };
    let identity =
        positions.len() == cfg.rows && positions.iter().enumerate().all(|(i, &p)| i == p);
    if !identity {
        let mut shape: Vec<usize> = rest.iter().map(|l| b.dims[l]).collect();
        shape.push(positions.len());
        let out_elems = shape.iter().product();
        let selected = b.index_select(slot, rest.len(), positions, out_elems);
        b.free_if_temp(slot);
        slot = selected;
    }
    let mut output_shape: Vec<usize> = rest.iter().map(|l| net.sizes[l]).collect();
    output_shape.push(n);

    // Peak of live temporaries, from the planned sizes.
    let mut live = 0usize;
    let mut peak = 0usize;
    for instr in &b.program {
        match instr {
            Instr::Free { slot } => live -= b.slot_elems[*slot],
            Instr::Einsum { out, .. }
            | Instr::IndexSelect { out, .. }
            | Instr::Reshape { out, .. }
            | Instr::Concatenate { out, .. } => {
                live += b.slot_elems[*out];
                peak = peak.max(live);
            }
        }
    }
    report.peak_elems = peak;

    Ok(SampledPlan {
        n_inputs,
        input_shapes,
        n_slots: b.slot_elems.len(),
        program: b.program,
        result: slot,
        output_shape,
        report,
    })
}
