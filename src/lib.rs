//! Prototype of a sampled einsum (sparse-state contraction) on tenferro,
//! for [tensor4all/tenferro-rs#2675](https://github.com/tensor4all/tenferro-rs/issues/2675).
//!
//! Given an einsum, a contraction path and a list of samples (values of some
//! output labels), compute
//!
//! ```text
//! Y[rest..., s] = T[rest..., sample s],   T = einsum(operands)
//! ```
//!
//! without forming `T`. Each tree node keeps one row per distinct partial
//! sample below it (artensor's sparse-state method). Execution is v0 of the
//! issue: only existing tenferro operations (`einsum`, `index_select`,
//! `reshape`, `concatenate`), so the eager layer differentiates through it.
//!
//! ```
//! use std::collections::HashMap;
//! use sampled_einsum::{plan, ConcreteOps, Network, PlanOptions, Samples};
//! use tenferro_cpu::CpuEngine;
//! use tenferro_tensor::{Tensor, TensorValue};
//!
//! // ij,jk->ik sampled at (i, k) = (0, 1) and (1, 2): SDDMM.
//! let net = Network {
//!     inputs: vec![vec![0, 1], vec![1, 2]],
//!     output: vec![0, 2],
//!     sizes: HashMap::from([(0, 2), (1, 3), (2, 4)]),
//! };
//! let samples = Samples { labels: vec![0, 2], n: 2, values: vec![0, 1, 1, 2] };
//! let p = plan(&net, &[(0, 1)], &samples, &PlanOptions::default())?;
//! let a = TensorValue::from_tensor(Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?);
//! let b = TensorValue::from_tensor(Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?);
//! let mut engine = CpuEngine::serial();
//! let mut ops = ConcreteOps::new(&mut engine);
//! let (y, _) = p.execute(&mut ops, &[&a, &b])?;
//! let y = ops.to_tensor(y)?;
//! assert_eq!(y.shape(), &[2]);
//! assert_eq!(y.as_slice::<f64>()?, &[3.0, 3.0]);
//! # Ok::<(), sampled_einsum::Error>(())
//! ```

mod exec;
mod plan;

pub use exec::{ConcreteOps, EagerOps, ExecStats, Ops};
pub use plan::{
    plan, Instr, Network, PlanOptions, PlanReport, SampledPlan, Samples, StepKind, StepReport,
};

/// Error type of the prototype: tenferro's errors and planning errors, boxed.
pub type Error = Box<dyn std::error::Error + 'static>;

/// Result type of the prototype.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err(message: impl Into<String>) -> Error {
    message.into().into()
}
