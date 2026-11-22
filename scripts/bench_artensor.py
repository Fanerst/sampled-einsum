"""Time artensor's sparse-state contraction of an exported network on the prototype's own tree.

Usage: python scripts/bench_artensor.py DATA_DIR ARTENSOR_SRC DTYPE DEVICE THREADS REPEATS

  DATA_DIR      a directory written by export_circuit.py (network.json, tensors.bin)
  ARTENSOR_SRC  an artensor checkout (put on sys.path; nothing is installed)
  DTYPE         complex64 or complex128
  DEVICE        cpu or cuda (the first visible CUDA device)
  THREADS       torch CPU threads
  REPEATS       timed runs of the same scheme

The script does not search for an order. It rebuilds artensor's contraction tree from the SSA
path in network.json (checking its complexity against the export), and compiles artensor's sparse
scheme from the exported depth-first order, for the same bitstrings and the same sc_target as the
export (30). That is the scheme artensor ran at export time, on the tree the prototype contracts.
Each run gets fresh tensors, since artensor's contraction overwrites its input list. Every run
prints a JSON summary line, with fidelities against the reference amplitudes and against the
artensor amplitudes recorded at export.
"""

import json
import resource
import sys
import time
from copy import deepcopy

data_dir, src, dtype_name, device, threads, repeats = sys.argv[1:7]
threads, repeats = int(threads), int(repeats)
sys.path.insert(0, src)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(threads)
from artensor import (  # noqa: E402
    AbstractTensorNetwork,
    ContractionTree,
    contraction_scheme_sparse,
    tensor_contraction_sparse,
)

SC_TARGET = 30
dtype = {"complex64": torch.complex64, "complex128": torch.complex128}[dtype_name]
dev = torch.device(device)

ex = json.load(open(f"{data_dir}/network.json"))
labels, outputs, bitstrings = ex["inputs"], ex["output"], ex["bitstrings"]
raw = np.fromfile(f"{data_dir}/tensors.bin", dtype="<c16")
tensors, offset = [], 0
for ls in labels:
    r = len(ls)
    # Each tensor is stored column-major in its label order: row-major over the reversed axes.
    t = raw[offset : offset + 2**r].reshape((2,) * r).transpose(tuple(reversed(range(r))))
    tensors.append(np.ascontiguousarray(t))
    offset += 2**r
assert offset == raw.size, "tensors.bin size"

# The network exactly as export_circuit.py handed it to artensor's find_order.
output_set = set(outputs)
n_labels = max(max(ls) for ls in labels) + 1
tensor_bonds = {i: [l for l in ls if l not in output_set] for i, ls in enumerate(labels)}
bond_dims = {l: 2.0 for l in range(n_labels) if l not in output_set}
final_nodes = [next(i for i, ls in enumerate(labels) if out in ls) for out in outputs]
tn = AbstractTensorNetwork(deepcopy(tensor_bonds), deepcopy(bond_dims), final_nodes, len(bitstrings))

# SSA path (step k creates node N + k) back to artensor's order, where (i, j) contracts the
# subtree holding leaf j into the one holding leaf i (the inverse of the export's conversion).
rep = list(range(len(labels)))
order = []
for a, b in ex["path"]:
    order.append((rep[a], rep[b]))
    rep.append(rep[a])
ctree = ContractionTree(tn, order, 0)
complexity = [float(x) for x in ctree.tree_complexity()]
assert all(abs(a - b) < 1e-9 for a, b in zip(complexity, ex["tree_complexity"])), (
    f"rebuilt tree complexity {complexity} differs from the export's {ex['tree_complexity']}"
)
# The rebuilt tree has the exported merges, but its own depth-first listing depends on the
# left/right orientation of each merge. artensor's scheme compiler takes its order from
# tree_order_dfs(), so hand it the exported order: the depth-first order artensor produced at
# export. The scheme is then the one artensor ran at export time.
ctree.tree_order_dfs = lambda: list(order)

t0 = time.time()
scheme, _, bitstrings_sorted = contraction_scheme_sparse(deepcopy(ctree), bitstrings, sc_target=SC_TARGET)
t_plan = time.time() - t0

reference = np.array([complex(re, im) for re, im in ex["reference"]])
exported = np.array([complex(re, im) for re, im in ex["artensor"]])
position = {b: k for k, b in enumerate(bitstrings_sorted)}


def fidelity(a, b):
    return float(abs(np.vdot(b, a)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real))


print(
    f"artensor tree rebuilt from the SSA path: tc 10^{complexity[0]:.2f}, sc 2^{complexity[1]:.1f}; "
    f"scheme {len(scheme)} steps in {t_plan:.2f} s; torch {torch.__version__}, {device}, {threads} threads, {dtype_name}",
    flush=True,
)

for k in range(1, repeats + 1):
    ts = [torch.from_numpy(t).to(dtype).to(dev) for t in tensors]
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = tensor_contraction_sparse(ts, scheme)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - t0
    amps_sorted = out.reshape(-1).cpu().numpy().astype(np.complex128)
    amps = np.array([amps_sorted[position[b]] for b in bitstrings])
    f_ref, f_exp = fidelity(amps, reference), fidelity(amps, exported)
    max_rel = float(np.max(np.abs(amps - reference)) / np.max(np.abs(reference)))
    summary = {
        "tool": "artensor",
        "device": device,
        "threads": threads,
        "dtype": dtype_name,
        "run": k,
        "seconds": seconds,
        "plan_seconds": t_plan,
        "fidelity": f_ref,
        "fidelity_vs_export": f_exp,
        "max_rel_diff": max_rel,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None,
        "torch": torch.__version__,
    }
    print(
        f"artensor {device} run {k}: {seconds:.2f} s; fidelity {f_ref:.12f} (vs export {f_exp:.12f}), "
        f"max |diff| / max |ref| = {max_rel:.2e}",
        flush=True,
    )
    print(json.dumps(summary), flush=True)
    del ts, out
