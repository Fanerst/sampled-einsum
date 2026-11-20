//! How long does dropping a large CPU tensor take, compared with a plain `Vec`?
//!
//! The n30 m14 run spent about 100 s of 257 s in dropping temporaries. This creates a
//! 2^30-element c128 tensor (16 GiB) with an outer-product einsum and times its drop,
//! next to the drop of a `Vec<Complex64>` of the same size.
//!
//! Usage: cargo run --release --example drop_cost -- [THREADS] [LOG2_ELEMS]

use std::time::Instant;

use num_complex::Complex64;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_einsum::EinsumOps;
use tenferro_tensor::Tensor;

fn main() -> sampled_einsum::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let threads: usize = args.get(1).map_or(Ok(16), |s| s.parse())?;
    let log2: u32 = args.get(2).map_or(Ok(30), |s| s.parse())?;
    let half = 1usize << (log2 / 2);
    let other = 1usize << (log2 - log2 / 2);
    let mut engine = CpuEngine::new(Pool::builder().threads(threads).build()?);
    let a = Tensor::from_vec_col_major(vec![half], vec![Complex64::new(1.0, 0.5); half])?;
    let b = Tensor::from_vec_col_major(vec![other], vec![Complex64::new(0.5, 1.0); other])?;
    for round in 0..3 {
        let t0 = Instant::now();
        let t = [&a, &b].einsum("i,j->ij", &mut engine)?;
        let create = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        drop(t);
        let tensor_drop = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        let v = vec![Complex64::new(1.0, 0.5); half * other];
        let fill = t2.elapsed().as_secs_f64();
        let t3 = Instant::now();
        drop(std::hint::black_box(v));
        let vec_drop = t3.elapsed().as_secs_f64();
        println!(
            "round {round}: 2^{log2} c128 ({:.0} GiB): tensor einsum {create:.2} s, tensor drop {tensor_drop:.3} s; Vec fill {fill:.2} s, Vec drop {vec_drop:.3} s",
            (half * other * 16) as f64 / (1u64 << 30) as f64
        );
    }
    Ok(())
}
