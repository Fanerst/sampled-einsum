//! Running a [`SampledPlan`] on tenferro: the concrete layer (`TensorValue` and an
//! engine) or the eager layer (`EagerSession`, which records gradients).

use std::collections::BTreeMap;
use std::time::Instant;

use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_einsum::{EagerEinsumOps, EinsumOps, EinsumSubscripts};
use tenferro_tensor::{
    Engine, GatherConfig, MemoryKind, Tensor, TensorRead, TensorValue, TensorView,
};

use crate::plan::{Instr, SampledPlan};
use crate::{err, Result};

/// The four tenferro operations a plan needs.
pub trait Ops {
    /// Tensor type of this layer.
    type T;
    /// Shape of a tensor.
    fn shape(&self, t: &Self::T) -> Vec<usize>;
    /// Einsum with integer labels.
    fn einsum(&mut self, inputs: &[&Self::T], subscripts: &EinsumSubscripts) -> Result<Self::T>;
    /// Select `positions` along `axis` (positions may repeat).
    fn index_select(&mut self, t: &Self::T, axis: usize, positions: &[usize]) -> Result<Self::T>;
    /// Column-major reshape.
    fn reshape(&mut self, t: &Self::T, shape: &[usize]) -> Result<Self::T>;
    /// Concatenate along `axis`.
    fn concatenate(&mut self, inputs: &[&Self::T], axis: usize) -> Result<Self::T>;
}

/// The concrete layer: tenferro values on any engine (CPU or CUDA).
///
/// Values are [`TensorValue`]s, an owned buffer with a layout that can differ from the
/// buffer's own, so a result computed with merged axes takes the plan's shape without a copy.
/// [`ConcreteOps::to_tensor`] turns a result into a compact [`Tensor`].
pub struct ConcreteOps<'e> {
    engine: &'e mut dyn Engine,
}

impl<'e> ConcreteOps<'e> {
    /// Wrap an engine (CPU or CUDA).
    pub fn new(engine: &'e mut dyn Engine) -> Self {
        Self { engine }
    }

    /// A compact tensor holding `value`: the value's own buffer when its layout is the
    /// buffer's, else a copy.
    pub fn to_tensor(&mut self, value: TensorValue) -> Result<Tensor> {
        if value.as_tensor().is_some() {
            return Ok(value.into_tensor()?);
        }
        let shape = value.shape().to_vec();
        Ok(self.engine.reshape_read(read(&value)?, &shape)?)
    }
}

/// A read of `value`.
fn read(value: &TensorValue) -> Result<TensorRead<'_>> {
    value
        .tensor_read()
        .ok_or_else(|| err("an external payload has no tensor read"))
}

/// `shape` with the axes before and after `axis` merged into one each.
fn around(shape: &[usize], axis: usize) -> [usize; 3] {
    [
        shape[..axis].iter().product(),
        shape[axis],
        shape[axis + 1..].iter().product(),
    ]
}

/// `value` read as `[prefix, extent, suffix]` around `axis`, which a compact column-major
/// layout takes as a relabeling, not a copy.
///
/// tenferro's CUDA gather and concatenation have fast kernels up to rank 8 only. Past that a
/// general kernel decodes every axis of every element with 64-bit division (and the gather
/// searches its window axes per axis), so a 30-axis gather of the circuit network ran about 30
/// times slower than the memory traffic it needs, the same at complex64 as at complex128.
fn merged_read(value: &TensorValue, axis: usize) -> Result<TensorRead<'_>> {
    let shape = around(value.shape(), axis);
    let view = value
        .tensor_view()
        .ok_or_else(|| err("an external payload has no tensor view"))?;
    // The dtype-erased view has no reshape of its own; each typed arm does.
    let view = match view {
        TensorView::F32(v) => TensorView::F32(v.reshape_view(&shape)?),
        TensorView::F64(v) => TensorView::F64(v.reshape_view(&shape)?),
        TensorView::I32(v) => TensorView::I32(v.reshape_view(&shape)?),
        TensorView::I64(v) => TensorView::I64(v.reshape_view(&shape)?),
        TensorView::Bool(v) => TensorView::Bool(v.reshape_view(&shape)?),
        TensorView::C32(v) => TensorView::C32(v.reshape_view(&shape)?),
        TensorView::C64(v) => TensorView::C64(v.reshape_view(&shape)?),
    };
    Ok(TensorRead::from_view(view))
}

/// Relabel a compact `[prefix, n, suffix]` result as `shape` with `n` on `axis`.
fn unmerge(t: Tensor, mut shape: Vec<usize>, axis: usize) -> Result<TensorValue> {
    shape[axis] = t.shape()[1];
    Ok(TensorValue::from_tensor(t).reshape_view(shape)?)
}

impl Ops for ConcreteOps<'_> {
    type T = TensorValue;

    fn shape(&self, t: &TensorValue) -> Vec<usize> {
        t.shape().to_vec()
    }

    fn einsum(
        &mut self,
        inputs: &[&TensorValue],
        subscripts: &EinsumSubscripts,
    ) -> Result<TensorValue> {
        let reads = inputs.iter().map(|v| read(v)).collect::<Result<Vec<_>>>()?;
        Ok(TensorValue::from_tensor(
            reads.einsum(subscripts, &mut *self.engine)?,
        ))
    }

    fn index_select(
        &mut self,
        t: &TensorValue,
        axis: usize,
        positions: &[usize],
    ) -> Result<TensorValue> {
        let [prefix, _, suffix] = around(t.shape(), axis);
        let config = GatherConfig {
            offset_dims: vec![0, 2],
            collapsed_slice_dims: vec![1],
            start_index_map: vec![1],
            index_vector_dim: 1,
            slice_sizes: vec![prefix, 1, suffix],
        };
        let index: Vec<i64> = positions
            .iter()
            .map(|&p| i64::try_from(p).map_err(|_| err(format!("position {p} exceeds i64"))))
            .collect::<Result<_>>()?;
        let host = Tensor::from_vec_col_major(vec![positions.len(), 1], index)?;
        let operand = merged_read(t, axis)?;
        // A device engine (CUDA) refuses host indices rather than copy them implicitly.
        let uploaded = match operand.placement().memory_kind {
            MemoryKind::Device => Some(
                self.engine
                    .upload_host_tensor(TensorRead::from_tensor(&host))?,
            ),
            _ => None,
        };
        let indices = TensorRead::from_tensor(uploaded.as_ref().unwrap_or(&host));
        let out = self.engine.gather_read(operand, indices, &config)?;
        unmerge(out, t.shape().to_vec(), axis)
    }

    fn reshape(&mut self, t: &TensorValue, shape: &[usize]) -> Result<TensorValue> {
        use tenferro_runtime::CoreOps;
        let out = match t.as_tensor() {
            Some(tensor) => tensor.reshape(shape, &mut *self.engine)?,
            None => self.engine.reshape_read(read(t)?, shape)?,
        };
        Ok(TensorValue::from_tensor(out))
    }

    fn concatenate(&mut self, inputs: &[&TensorValue], axis: usize) -> Result<TensorValue> {
        let first = inputs
            .first()
            .ok_or_else(|| err("concatenate needs at least one input"))?;
        let reads = inputs
            .iter()
            .map(|v| merged_read(v, axis))
            .collect::<Result<Vec<_>>>()?;
        let out = self.engine.concatenate_read(&reads, 1)?;
        unmerge(out, first.shape().to_vec(), axis)
    }
}

/// The eager layer: operations run immediately and are recorded for reverse mode.
pub struct EagerOps<'s, 'a> {
    session: &'s mut EagerSession<'a>,
}

impl<'s, 'a> EagerOps<'s, 'a> {
    /// Wrap a borrowed eager session.
    pub fn new(session: &'s mut EagerSession<'a>) -> Self {
        Self { session }
    }
}

impl Ops for EagerOps<'_, '_> {
    type T = EagerTensor;

    fn shape(&self, t: &EagerTensor) -> Vec<usize> {
        t.shape().to_vec()
    }

    fn einsum(
        &mut self,
        inputs: &[&EagerTensor],
        subscripts: &EinsumSubscripts,
    ) -> Result<EagerTensor> {
        Ok(self.session.einsum(inputs, subscripts)?)
    }

    fn index_select(
        &mut self,
        t: &EagerTensor,
        axis: usize,
        positions: &[usize],
    ) -> Result<EagerTensor> {
        Ok(self.session.index_select(t, axis as isize, positions)?)
    }

    fn reshape(&mut self, t: &EagerTensor, shape: &[usize]) -> Result<EagerTensor> {
        Ok(self.session.reshape(t, shape.to_vec())?)
    }

    fn concatenate(&mut self, inputs: &[&EagerTensor], axis: usize) -> Result<EagerTensor> {
        Ok(self.session.concatenate(inputs, axis as isize)?)
    }
}

/// What one execution did.
#[derive(Clone, Debug, Default)]
pub struct ExecStats {
    /// Largest total element count of live temporaries, measured on the actual tensors.
    pub peak_live_elems: usize,
    /// Instructions run.
    pub instructions: usize,
    /// Wall time per operation kind, in seconds. On a device engine this measures
    /// submission rather than completion.
    pub seconds: BTreeMap<&'static str, f64>,
}

fn slot<'a, T>(operands: &[&'a T], temps: &'a [Option<T>], s: usize) -> Result<&'a T> {
    if s < operands.len() {
        Ok(operands[s])
    } else {
        temps[s]
            .as_ref()
            .ok_or_else(|| err(format!("slot {s} read before it was written")))
    }
}

impl SampledPlan {
    /// Run the plan on `operands` (in network order) and return the `[rest..., n]` result.
    pub fn execute<O: Ops>(&self, ops: &mut O, operands: &[&O::T]) -> Result<(O::T, ExecStats)> {
        if operands.len() != self.n_inputs {
            return Err(err(format!(
                "expected {} operands, got {}",
                self.n_inputs,
                operands.len()
            )));
        }
        for (i, (t, want)) in operands.iter().zip(&self.input_shapes).enumerate() {
            let got = ops.shape(t);
            if &got != want {
                return Err(err(format!(
                    "operand {i} has shape {got:?}, the network says {want:?}"
                )));
            }
        }
        let mut temps: Vec<Option<O::T>> = (0..self.n_slots).map(|_| None).collect();
        let numel = |shape: Vec<usize>| -> usize { shape.iter().product() };
        let mut live = 0usize;
        let mut stats = ExecStats::default();
        for instr in &self.program {
            let started = Instant::now();
            let (kind, out, value) = match instr {
                Instr::Einsum {
                    inputs,
                    input_labels,
                    output_labels,
                    out,
                } => {
                    let refs = inputs
                        .iter()
                        .map(|&s| slot(operands, &temps, s))
                        .collect::<Result<Vec<_>>>()?;
                    let labels: Vec<&[u32]> = input_labels.iter().map(Vec::as_slice).collect();
                    let subscripts = EinsumSubscripts::new(&labels, output_labels);
                    ("einsum", *out, ops.einsum(&refs, &subscripts)?)
                }
                Instr::IndexSelect {
                    input,
                    axis,
                    positions,
                    out,
                } => {
                    let t = slot(operands, &temps, *input)?;
                    ("index_select", *out, ops.index_select(t, *axis, positions)?)
                }
                Instr::Reshape { input, shape, out } => {
                    let t = slot(operands, &temps, *input)?;
                    ("reshape", *out, ops.reshape(t, shape)?)
                }
                Instr::Concatenate { inputs, axis, out } => {
                    let refs = inputs
                        .iter()
                        .map(|&s| slot(operands, &temps, s))
                        .collect::<Result<Vec<_>>>()?;
                    ("concatenate", *out, ops.concatenate(&refs, *axis)?)
                }
                Instr::Free { slot } => {
                    if let Some(t) = temps[*slot].take() {
                        live -= numel(ops.shape(&t));
                        drop(t);
                    }
                    *stats.seconds.entry("free").or_default() += started.elapsed().as_secs_f64();
                    stats.instructions += 1;
                    continue;
                }
            };
            *stats.seconds.entry(kind).or_default() += started.elapsed().as_secs_f64();
            live += numel(ops.shape(&value));
            stats.peak_live_elems = stats.peak_live_elems.max(live);
            temps[out] = Some(value);
            stats.instructions += 1;
        }
        let result = temps[self.result]
            .take()
            .ok_or_else(|| err("the plan produced no result"))?;
        let got = ops.shape(&result);
        if got != self.output_shape {
            return Err(err(format!(
                "result has shape {got:?}, expected {:?}",
                self.output_shape
            )));
        }
        Ok((result, stats))
    }
}
