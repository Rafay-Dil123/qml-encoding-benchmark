"""qenc.py - shared code for the data-encoding benchmark.

Data pipeline (fixed stratified folds, leakage-free preprocessing), three data
encodings (angle, IQP/ZZ, amplitude) in front of one shared trainable ansatz,
resource counting, training, and a gate-level noise model.
"""
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one thread per worker; runs are parallelised instead

import numpy as onp
import pandas as pd
import pennylane as qml
from joblib import Parallel, delayed
from pennylane import numpy as pnp
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATASETS = ("iris", "wine", "breast_cancer")
ENCODINGS = ("angle", "zz", "amplitude")
N_FOLDS = 5
FOLD_SEED = 42
BASIS_GATES = {"RX", "RY", "RZ", "Hadamard", "CNOT"}


# --------------------------------------------------------------------------
# 1. Data and fixed folds
# --------------------------------------------------------------------------
def load_dataset(name):
    """Return (X, y) as float / int arrays for one of the three datasets."""
    loader = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}[name]
    d = loader()
    return d.data.astype(float), d.target.astype(int)


def make_folds(y, n_folds=N_FOLDS, seed=FOLD_SEED):
    """Stratified k-fold test-index lists; deterministic for a given seed."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [te.tolist() for _, te in skf.split(onp.zeros(len(y)), y)]


def save_folds(path="folds/folds.json"):
    """Create the fold file once. Every experiment reads the same file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {}
    for name in DATASETS:
        _, y = load_dataset(name)
        out[name] = {"n_samples": int(len(y)), "n_folds": N_FOLDS, "seed": FOLD_SEED,
                     "test_indices": make_folds(y)}
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def load_folds(path="folds/folds.json"):
    with open(path) as f:
        return json.load(f)


def get_split(name, fold, folds):
    """Raw (unscaled) train/test split for one fixed fold."""
    X, y = load_dataset(name)
    te = onp.array(folds[name]["test_indices"][fold])
    mask = onp.ones(len(y), dtype=bool)
    mask[te] = False
    return X[mask], y[mask], X[te], y[te]


def reduce_features(Xtr, Xte, n_features):
    """Standardise, then PCA to n_features if the data has more. Fit on train only."""
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    if n_features < Ztr.shape[1]:
        pca = PCA(n_components=n_features, random_state=0).fit(Ztr)
        Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
    return Ztr, Zte


def features_for_encoding(encoding, n_qubits):
    """Number of classical features an encoding loads into n_qubits."""
    return 2 ** n_qubits if encoding == "amplitude" else n_qubits


def prepare(Xtr, Xte, encoding, n_qubits, scale=onp.pi, amp_mode="minmax"):
    """Encoding-specific preprocessing. All statistics come from the training fold.

    angle / zz : PCA to n_qubits features, min-max to [0, scale] (test clipped).
    amplitude  : PCA to at most 2**n_qubits features, rescale, zero-pad and
                 L2-normalise each sample. amp_mode picks the rescaling:
                   "minmax"      min-max to [0, 1]  (default; non-negative amplitudes)
                   "standardize" zero mean, unit variance (signed amplitudes)
                   "bias"        one feature fewer, standardised, plus a constant 1
    """
    if encoding in ("angle", "zz"):
        Ztr, Zte = reduce_features(Xtr, Xte, n_qubits)
        mm = MinMaxScaler(feature_range=(0.0, scale)).fit(Ztr)
        return mm.transform(Ztr), onp.clip(mm.transform(Zte), 0.0, scale)
    if encoding == "amplitude":
        dim = 2 ** n_qubits
        n_feat = dim - 1 if amp_mode == "bias" else dim
        Ztr, Zte = reduce_features(Xtr, Xte, min(n_feat, Xtr.shape[1]))
        if amp_mode == "minmax":
            mm = MinMaxScaler().fit(Ztr)
            parts = [mm.transform(Ztr), onp.clip(mm.transform(Zte), 0.0, 1.0)]
        else:
            sc = StandardScaler().fit(Ztr)
            parts = [sc.transform(Ztr), sc.transform(Zte)]
            if amp_mode == "bias":
                parts = [onp.hstack([Z, onp.ones((Z.shape[0], 1))]) for Z in parts]
        out = []
        for Z in parts:
            if Z.shape[1] < dim:
                Z = onp.hstack([Z, onp.zeros((Z.shape[0], dim - Z.shape[1]))])
            norm = onp.linalg.norm(Z, axis=1, keepdims=True)
            dead = norm[:, 0] < 1e-12            # all-zero sample: use the uniform state
            Z[dead], norm[dead] = 1.0, onp.sqrt(dim)
            out.append(Z / norm)
        return out[0], out[1]
    raise ValueError(encoding)


# --------------------------------------------------------------------------
# 2. Circuits: three encodings, one ansatz
# --------------------------------------------------------------------------
def zz_pairs(n_qubits, entanglement="full"):
    if entanglement == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    return [(i, i + 1) for i in range(n_qubits - 1)]


def apply_encoding(x, encoding, n_qubits, zz_reps=2, zz_entanglement="full", mottonen=False):
    """Apply the data-encoding block. x has shape (..., n_features)."""
    wires = list(range(n_qubits))
    if encoding == "angle":
        for i in wires:
            qml.RY(x[..., i], wires=i)
    elif encoding == "zz":
        # ZZ feature map of Havlicek et al. (2019), Qiskit ZZFeatureMap convention.
        for _ in range(zz_reps):
            for i in wires:
                qml.Hadamard(wires=i)
                qml.RZ(2.0 * x[..., i], wires=i)
            for i, j in zz_pairs(n_qubits, zz_entanglement):
                qml.CNOT(wires=[i, j])
                qml.RZ(2.0 * (onp.pi - x[..., i]) * (onp.pi - x[..., j]), wires=j)
                qml.CNOT(wires=[i, j])
    elif encoding == "amplitude":
        if mottonen:   # explicit gate-level state preparation (single sample)
            qml.MottonenStatePreparation(x, wires=wires)
        else:          # same state, written directly into the simulator
            qml.StatePrep(x, wires=wires)
    else:
        raise ValueError(encoding)


def apply_ansatz(weights, n_qubits, entangle=True):
    """Shared trainable ansatz. weights has shape (layers, n_qubits, 3).

    Each layer: a general single-qubit rotation RZ-RY-RZ on every qubit,
    then a ring of CNOTs (a single CNOT for two qubits).
    """
    for layer in range(weights.shape[0]):
        for q in range(n_qubits):
            qml.RZ(weights[layer, q, 0], wires=q)
            qml.RY(weights[layer, q, 1], wires=q)
            qml.RZ(weights[layer, q, 2], wires=q)
        if entangle and n_qubits > 1:
            if n_qubits == 2:
                qml.CNOT(wires=[0, 1])
            else:
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])


def readout_wires(n_classes):
    """Class c is read out as basis state |c> of the first ceil(log2 C) qubits."""
    return list(range(int(onp.ceil(onp.log2(n_classes)))))


def make_qnode(encoding, n_qubits, n_classes, entangle=True, **enc_kw):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(weights, x):
        apply_encoding(x, encoding, n_qubits, **enc_kw)
        apply_ansatz(weights, n_qubits, entangle)
        return qml.probs(wires=readout_wires(n_classes))

    return circuit


def class_probs(raw, n_classes):
    """Keep the first n_classes outcomes and renormalise."""
    p = raw[..., :n_classes]
    return p / (pnp.sum(p, axis=-1, keepdims=True) + 1e-12)


# --------------------------------------------------------------------------
# 3. Resource counting
# --------------------------------------------------------------------------
def _decomposed_tape(encoding, n_qubits, layers, x, entangle=True, with_ansatz=True, **enc_kw):
    w = onp.full((layers, n_qubits, 3), 0.3)
    with qml.tape.QuantumTape() as tape:
        apply_encoding(x, encoding, n_qubits, mottonen=True, **enc_kw)
        if with_ansatz:
            apply_ansatz(w, n_qubits, entangle)
    (tape,), _ = qml.transforms.decompose(tape, gate_set=BASIS_GATES)
    ops = [op for op in tape.operations if op.name != "GlobalPhase"]
    return qml.tape.QuantumTape(ops, [])


def count_resources(encoding, n_qubits, layers, x_example, entangle=True, **enc_kw):
    """Gate counts in the basis {RX, RY, RZ, H, CNOT} for one data point."""
    out = {"encoding": encoding, "n_qubits": n_qubits, "layers": layers,
           "n_features": features_for_encoding(encoding, n_qubits),
           "n_params": layers * n_qubits * 3}
    for label, with_ansatz in (("enc", False), ("total", True)):
        t = _decomposed_tape(encoding, n_qubits, layers, x_example, entangle, with_ansatz, **enc_kw)
        two = sum(1 for op in t.operations if len(op.wires) == 2)
        out[f"{label}_1q"] = len(t.operations) - two
        out[f"{label}_2q"] = two
        out[f"{label}_depth"] = t.graph.get_depth()
    return out


# --------------------------------------------------------------------------
# 4. Training
# --------------------------------------------------------------------------
def train_vqc(Xtr, ytr, Xte, yte, encoding, n_qubits, n_classes, layers=2, seed=0,
              epochs=100, lr=0.05, entangle=True, **enc_kw):
    """Full-batch Adam on the cross-entropy of the read-out distribution."""
    circuit = make_qnode(encoding, n_qubits, n_classes, entangle, **enc_kw)
    rng = onp.random.default_rng(seed)
    w = pnp.array(rng.uniform(-onp.pi, onp.pi, size=(layers, n_qubits, 3)), requires_grad=True)
    Xtr_, Xte_ = pnp.array(Xtr, requires_grad=False), pnp.array(Xte, requires_grad=False)
    onehot = onp.eye(n_classes)[ytr]

    def loss_fn(weights):
        p = class_probs(circuit(weights, Xtr_), n_classes)
        return -pnp.mean(pnp.sum(onehot * pnp.log(p + 1e-10), axis=1))

    grad_fn = qml.grad(loss_fn)
    m, v = onp.zeros(w.shape), onp.zeros(w.shape)
    b1, b2, eps = 0.9, 0.999, 1e-8
    losses, grad0 = [], None
    t0 = time.time()
    for step in range(1, epochs + 1):
        g = onp.asarray(grad_fn(w))
        if step == 1:
            grad0 = float(onp.linalg.norm(g))
        losses.append(float(loss_fn(w)))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w = pnp.array(w - lr * (m / (1 - b1 ** step)) / (onp.sqrt(v / (1 - b2 ** step)) + eps),
                      requires_grad=True)
    losses.append(float(loss_fn(w)))

    def predict(X):
        return onp.argmax(onp.asarray(class_probs(circuit(w, X), n_classes)), axis=1)

    return {"weights": onp.asarray(w), "losses": losses, "grad_norm_init": grad0,
            "train_acc": float(onp.mean(predict(Xtr_) == ytr)),
            "test_acc": float(onp.mean(predict(Xte_) == yte)),
            "train_time_s": time.time() - t0}


def run_config(dataset, fold, folds, encoding, n_qubits, layers=2, seed=0, epochs=100, lr=0.05,
               scale=onp.pi, entangle=True, amp_mode="minmax", keep_model=False, **enc_kw):
    """One training run on one fixed fold. Returns a flat result record."""
    Xtr, ytr, Xte, yte = get_split(dataset, fold, folds)
    n_classes = int(len(onp.unique(ytr)))
    Ptr, Pte = prepare(Xtr, Xte, encoding, n_qubits, scale, amp_mode)
    res = train_vqc(Ptr, ytr, Pte, yte, encoding, n_qubits, n_classes, layers, seed,
                    epochs, lr, entangle, **enc_kw)
    rec = {"dataset": dataset, "fold": fold, "encoding": encoding, "n_qubits": n_qubits,
           "layers": layers, "seed": seed, "epochs": epochs, "lr": lr, "scale": scale,
           "entangle": entangle, "amp_mode": amp_mode, "zz_reps": enc_kw.get("zz_reps", 2),
           "zz_entanglement": enc_kw.get("zz_entanglement", "full"),
           "train_acc": res["train_acc"], "test_acc": res["test_acc"],
           "final_loss": res["losses"][-1], "grad_norm_init": res["grad_norm_init"],
           "train_time_s": res["train_time_s"], "losses": json.dumps(res["losses"])}
    if keep_model:
        rec["weights"] = json.dumps(res["weights"].tolist())
    return rec


def _run_chunks(jobs, path, n_jobs=2, chunk=16):
    """Run delayed jobs in parallel, appending finished chunks to `path + ".part"`.

    An interrupted grid resumes from the last finished chunk; the job order is
    deterministic, so the number of saved rows identifies where to continue.
    """
    part = path + ".part"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n_done = len(pd.read_csv(part)) if os.path.exists(part) else 0
    for i in range(n_done, len(jobs), chunk):
        recs = Parallel(n_jobs=n_jobs)(jobs[i:i + chunk])
        pd.DataFrame(recs).to_csv(part, mode="a", header=(i == 0), index=False)
    os.replace(part, path)
    return pd.read_csv(path)


def run_grid(configs, folds, path, n_jobs=2, force=False):
    """Run a list of run_config keyword dicts in parallel; cache the table as CSV."""
    if os.path.exists(path) and not force:
        return pd.read_csv(path)
    return _run_chunks([delayed(run_config)(folds=folds, **c) for c in configs], path, n_jobs)


# --------------------------------------------------------------------------
# 5. Noise model
# --------------------------------------------------------------------------
def noisy_probs(weights, x, encoding, n_qubits, n_classes, p1, p2, p_meas, entangle=True, **enc_kw):
    """Read-out distribution of one sample under gate-level noise (density matrix).

    The circuit is first compiled to {RX, RY, RZ, H, CNOT}; amplitude encoding
    is compiled with the Mottonen state preparation. A depolarising channel of
    strength p1 follows every one-qubit gate, one of strength p2 acts on both
    qubits after every CNOT, and a bit flip of probability p_meas precedes
    measurement of each read-out qubit.
    """
    with qml.tape.QuantumTape() as tape:
        apply_encoding(x, encoding, n_qubits, mottonen=True, **enc_kw)
        apply_ansatz(weights, n_qubits, entangle)
    (tape,), _ = qml.transforms.decompose(tape, gate_set=BASIS_GATES)
    ops = [op for op in tape.operations if op.name != "GlobalPhase"]
    ro = readout_wires(n_classes)
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        for op in ops:
            qml.apply(op)
            p = p1 if len(op.wires) == 1 else p2
            if p > 0:
                for wq in op.wires:
                    qml.DepolarizingChannel(p, wires=wq)
        if p_meas > 0:
            for wq in ro:
                qml.BitFlip(p_meas, wires=wq)
        return qml.probs(wires=ro)

    return onp.asarray(circuit())


BASE_NOISE = {"p1": 1e-3, "p2": 1e-2, "p_meas": 2e-2}   # 1q gate, CNOT, read-out


def noisy_eval(rec, folds, lam, shots=1024):
    """Test accuracy of one trained model (a row of the main results table)
    when every error rate in BASE_NOISE is multiplied by lam.

    Returns the accuracy from exact noisy probabilities and from `shots` samples.
    """
    Xtr, ytr, Xte, yte = get_split(rec["dataset"], int(rec["fold"]), folds)
    n_classes, nq = int(len(onp.unique(ytr))), int(rec["n_qubits"])
    _, Pte = prepare(Xtr, Xte, rec["encoding"], nq, float(rec["scale"]), rec["amp_mode"])
    w = onp.array(json.loads(rec["weights"]))
    rng = onp.random.default_rng(1000 * int(rec["seed"]) + int(rec["fold"]))
    hit_exact = hit_shots = 0
    for x, y in zip(Pte, yte):
        p = noisy_probs(w, x, rec["encoding"], nq, n_classes, lam * BASE_NOISE["p1"],
                        lam * BASE_NOISE["p2"], lam * BASE_NOISE["p_meas"],
                        zz_reps=int(rec["zz_reps"]), zz_entanglement=rec["zz_entanglement"])
        p = onp.clip(p.real, 0, None)
        p = p / p.sum()
        hit_exact += int(onp.argmax(p[:n_classes]) == y)
        counts = rng.multinomial(shots, p)
        hit_shots += int(onp.argmax(counts[:n_classes]) == y)
    return {"dataset": rec["dataset"], "fold": int(rec["fold"]), "encoding": rec["encoding"],
            "seed": int(rec["seed"]), "lam": lam, "shots": shots,
            "acc_noisy": hit_exact / len(yte), "acc_noisy_shots": hit_shots / len(yte),
            "acc_ideal": float(rec["test_acc"])}


def noise_grid(df_models, folds, lams, path, n_jobs=2, force=False):
    if os.path.exists(path) and not force:
        return pd.read_csv(path)
    rows = [r for _, r in df_models.iterrows()]
    return _run_chunks([delayed(noisy_eval)(r, folds, lam) for r in rows for lam in lams], path, n_jobs)


# --------------------------------------------------------------------------
# 6. Plot style shared by both notebooks
# --------------------------------------------------------------------------
COLORS = {"angle": "#2a78d6", "zz": "#eb6834", "amplitude": "#1baf7a",
          "logreg": "#52514e", "svm_rbf": "#a3a29b"}
LABELS = {"angle": "Angle", "zz": "IQP/ZZ", "amplitude": "Amplitude",
          "logreg": "Logistic regression", "svm_rbf": "RBF SVM",
          "iris": "Iris", "wine": "Wine", "breast_cancer": "Breast Cancer"}


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": "#52514e", "ytick.color": "#52514e",
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
    })
