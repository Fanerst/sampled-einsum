"""Export artensor's n30 m14 Sycamore circuit as a sparse-state contraction for the Rust prototype.

Builds the tensor network from `circuit_n30_m14_s0_e0_pEFGH.py` without cirq (the five
gate types it uses are written out below), finds a sparse-state-aware order with
artensor's `find_order`, checks artensor's own sparse-state amplitudes against the
reference amplitudes, and writes:

  OUT_DIR/network.json  labels, SSA path, bitstrings, reference and artensor amplitudes
  OUT_DIR/tensors.bin   every tensor as little-endian complex128, column-major, in node order

Usage: python export_circuit.py CIRCUIT_PY AMPLITUDES_TXT OUT_DIR N_BITSTRINGS SC_TARGET SEED
"""

import json
import re
import sys
import time
from copy import deepcopy

import numpy as np
import torch

from artensor import contraction_scheme_sparse, find_order, tensor_contraction_sparse

circuit_py, amplitudes_txt, out_dir = sys.argv[1:4]
n_bitstrings, sc_target, seed = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

# ---- gates (cirq conventions) ----


def xpow(t):
    g, c, s = np.exp(0.5j * np.pi * t), np.cos(0.5 * np.pi * t), np.sin(0.5 * np.pi * t)
    return g * np.array([[c, -1j * s], [-1j * s, c]])


def ypow(t):
    g, c, s = np.exp(0.5j * np.pi * t), np.cos(0.5 * np.pi * t), np.sin(0.5 * np.pi * t)
    return g * np.array([[c, -s], [s, c]])


def zpow(p):
    return np.diag([1.0, np.exp(1j * np.pi * p)])


def phased_x(p, t):
    # cirq.PhasedXPowGate: the circuit Z^-p, X^t, Z^p.
    return zpow(p) @ xpow(t) @ zpow(-p)


def rz(rads):
    return np.diag([np.exp(-0.5j * rads), np.exp(0.5j * rads)])


def fsim(theta, phi):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, np.exp(-1j * phi)]])


# ---- parse the cirq circuit text ----

src = open(circuit_py).read()
head, body = src.split("CIRCUIT = ")
qubits = re.findall(r"cirq\.GridQubit\((\d+), (\d+)\)", head)
index = {q: k for k, q in enumerate(qubits)}
n = len(qubits)
text = re.sub(r"\s+", " ", body)
num = r"(-?[0-9.]+(?:e-?[0-9]+)?)"
gq = r"cirq\.GridQubit\((\d+), (\d+)\)"
pattern = re.compile(
    rf"cirq\.PhasedXPowGate\(phase_exponent={num}, exponent={num}\)\.on\({gq}\)"
    rf"|\(cirq\.X\*\*{num}\)\.on\({gq}\)"
    rf"|\(cirq\.Y\*\*{num}\)\.on\({gq}\)"
    rf"|cirq\.Rz\(np\.pi \* {num}\)\.on\({gq}\)"
    rf"|cirq\.FSimGate\(theta={num}, phi={num}\)\.on\( ?{gq}, {gq}\)"
)
gates = []  # (kind, matrix, qubits)
for m in pattern.finditer(text):
    g = m.groups()
    if g[0] is not None:
        gates.append(("1q", phased_x(float(g[0]), float(g[1])), (index[(g[2], g[3])],)))
    elif g[4] is not None:
        gates.append(("1q", xpow(float(g[4])), (index[(g[5], g[6])],)))
    elif g[7] is not None:
        gates.append(("1q", ypow(float(g[7])), (index[(g[8], g[9])],)))
    elif g[10] is not None:
        gates.append(("1q", rz(np.pi * float(g[10])), (index[(g[11], g[12])],)))
    else:
        gates.append(("2q", fsim(float(g[13]), float(g[14])), (index[(g[15], g[16])], index[(g[17], g[18])])))
counts = {k: sum(1 for x in gates if x[0] == k) for k in ("1q", "2q")}
written = {k: len(re.findall(p, text)) for k, p in [("1q", r"PhasedXPowGate|cirq\.X\*\*|cirq\.Y\*\*|cirq\.Rz"), ("2q", r"FSimGate")]}
assert counts == written, (counts, written)
print(f"{n} qubits, {counts['1q']} one-qubit and {counts['2q']} two-qubit gates", flush=True)

# ---- tensor network: |0> per qubit, one node per fSim, one final node per qubit ----
# Single-qubit gates on a wire are folded into the next fSim on it, or into the final node.

tensors, labels = [], []
next_label = 0


def new_label():
    global next_label
    next_label += 1
    return next_label - 1


wire = []
for q in range(n):
    wire.append(new_label())
    tensors.append(np.array([1.0, 0.0], dtype=complex))
    labels.append([wire[q]])
pending = [np.eye(2, dtype=complex) for _ in range(n)]
for kind, u, qs in gates:
    if kind == "1q":
        (q,) = qs
        pending[q] = u @ pending[q]
        continue
    q1, q2 = qs
    u = u @ np.kron(pending[q1], pending[q2])
    pending[q1] = np.eye(2, dtype=complex)
    pending[q2] = np.eye(2, dtype=complex)
    out1, out2 = new_label(), new_label()
    # T[in1, in2, out1, out2] = U[out1 * 2 + out2, in1 * 2 + in2]
    tensors.append(u.reshape(2, 2, 2, 2).transpose(2, 3, 0, 1).copy())
    labels.append([wire[q1], wire[q2], out1, out2])
    wire[q1], wire[q2] = out1, out2
final_nodes = []
outputs = []
for q in range(n):
    out = new_label()
    outputs.append(out)
    final_nodes.append(len(tensors))
    # F[bit, in] = <bit| pending |in>: axis 0 is artensor's configuration axis.
    tensors.append(pending[q].copy())
    labels.append([out, wire[q]])
print(f"{len(tensors)} nodes, {next_label} labels", flush=True)

# ---- artensor: order and its own sparse-state amplitudes ----

samples = []
with open(amplitudes_txt) as f:
    for line in f:
        b, re_, im_ = line.split()
        samples.append((b, float(re_) + 1j * float(im_)))
samples = samples[:n_bitstrings]
bitstrings = [b for b, _ in samples]
reference = np.array([a for _, a in samples])

output_set = set(outputs)
tensor_bonds = {i: [l for l in ls if l not in output_set] for i, ls in enumerate(labels)}
bond_dims = {l: 2.0 for l in range(next_label) if l not in output_set}
t0 = time.time()
order, slicing_bonds, ctree = find_order(
    tensor_bonds, bond_dims, final_nodes, seed, len(bitstrings),
    sc_target=sc_target, trials=4, iters=10, slicing_repeat=1, betas=np.linspace(3.0, 21.0, 61),
)
print(f"order found in {time.time() - t0:.1f} s; sliced bonds: {slicing_bonds}", flush=True)
print("tree complexity (tc log10, sc log2, ...):", ctree.tree_complexity(), flush=True)
assert not slicing_bonds, "raise SC_TARGET: the prototype does not slice"

dfs = ctree.tree_order_dfs()
t0 = time.time()
scheme, _, bitstrings_sorted = contraction_scheme_sparse(deepcopy(ctree), bitstrings, sc_target=sc_target)
t_plan = time.time() - t0
art_tensors = [torch.from_numpy(t).to(torch.complex64) for t in tensors]
t0 = time.time()
amps_sorted = tensor_contraction_sparse(art_tensors, scheme).reshape(-1).numpy()
t_run = time.time() - t0
by_string = dict(zip(bitstrings_sorted, amps_sorted))
artensor_amps = np.array([by_string[b] for b in bitstrings])
overlap = abs(np.vdot(reference, artensor_amps)) ** 2 / (np.vdot(reference, reference).real * np.vdot(artensor_amps, artensor_amps).real)
rel = np.max(np.abs(artensor_amps - reference)) / np.max(np.abs(reference))
print(f"artensor sparse-state: plan {t_plan:.2f} s, contraction {t_run:.2f} s (torch {torch.__version__}, CPU, {torch.get_num_threads()} threads)")
print(f"artensor vs reference: fidelity {overlap:.12f}, max |diff| / max |ref| = {rel:.2e}", flush=True)

# ---- export ----

# artensor contracts (i, j) into node i; convert to an SSA path where step k creates node N + k.
current = list(range(len(tensors)))
path = []
for k, (i, j) in enumerate(dfs):
    path.append([current[i], current[j]])
    current[i] = len(tensors) + k

with open(f"{out_dir}/tensors.bin", "wb") as f:
    for t in tensors:
        # Column-major data for the label order: the row-major data of the reversed axes.
        f.write(np.ascontiguousarray(t.transpose(tuple(reversed(range(t.ndim))))).astype("<c16").tobytes())
json.dump(
    {
        "inputs": labels,
        "output": outputs,
        "path": path,
        "bitstrings": bitstrings,
        "reference": [[a.real, a.imag] for a in reference],
        "artensor": [[complex(a).real, complex(a).imag] for a in artensor_amps],
        "artensor_seconds": {"plan": t_plan, "contraction": t_run, "threads": torch.get_num_threads()},
        "tree_complexity": [float(x) for x in ctree.tree_complexity()],
    },
    open(f"{out_dir}/network.json", "w"),
)
print(f"wrote {out_dir}/network.json and tensors.bin")
