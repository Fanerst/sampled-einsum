//! Running a [`SampledPlan`] on tenferro: the concrete layer (`Tensor` and an
//! engine) or the eager layer (`EagerSession`, which records gradients).

use std::collections::BTreeMap;
use std::time::Instant;

use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_einsum::{EagerEinsumOps, EinsumOps, EinsumSubscripts};
use tenferro_tensor::{Engine, Tensor};

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

/// The concrete layer: `Tensor` values on any tenferro engine.
pub struct ConcreteOps<'e> {
    engine: &'e mut dyn Engine,
}

impl<'e> ConcreteOps<'e> {
    /// Wrap an engine (CPU or CUDA).
    pub fn new(engine: &'e mut dyn Engine) -> Self {
        Self { engine }
    }
}

impl Ops for ConcreteOps<'_> {
    type T = Tensor;

    fn shape(&self, t: &Tensor) -> Vec<usize> {
        t.shape().to_vec()
    }

    fn einsum(&mut self, inputs: &[&Tensor], subscripts: &EinsumSubscripts) -> Result<Tensor> {
        Ok(inputs.einsum(subscripts, &mut *self.engine)?)
    }

    fn index_select(&mut self, t: &Tensor, axis: usize, positions: &[usize]) -> Result<Tensor> {
        Ok(t.index_select(axis as isize, positions, &mut *self.engine)?)
    }

    fn reshape(&mut self, t: &Tensor, shape: &[usize]) -> Result<Tensor> {
        use tenferro_runtime::CoreOps;
        Ok(t.reshape(shape, &mut *self.engine)?)
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> Result<Tensor> {
        Ok(self.engine.concatenate(inputs, axis)?)
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
