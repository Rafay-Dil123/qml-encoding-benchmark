# Which data encoding gives the best accuracy per qubit and per gate?

Angle, IQP/ZZ and amplitude encoding are compared in front of one shared trainable ansatz on Iris, Wine and Breast Cancer Wisconsin. For every configuration the qubits, circuit depth and two-qubit gates are counted, and accuracy and training stability are measured over fixed cross-validation folds and several seeds. Logistic regression and an RBF SVM on the same folds are the classical references.

Author: Rafay Dil (25280068)

## Deliverables

| Deliverable | Notebook | Report |
|---|---|---|
| 1. Data pipeline, fixed folds, classical baselines, first quantum model | `01_pipeline_baselines_first_model.ipynb` | `reports/Report1_Introduction_and_Methods.pdf` |
| 2. Full simulator results, noise model, ablations | `02_full_results_noise_ablations.ipynb` | `reports/Report2_Results_Draft.pdf` |

## Layout

```
01_pipeline_baselines_first_model.ipynb   deliverable 1 (also writes qenc.py)
02_full_results_noise_ablations.ipynb     deliverable 2
qenc.py                                   shared module: data, folds, encodings, ansatz, training, noise model
folds/folds.json                          the fixed stratified 5-fold splits used by every model
results/                                  every table behind the reports, as CSV (one row per training run)
figures/                                  figures used in the reports
reports/                                  the two PDF reports and the scripts that build them
```

## Reproduce

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_pipeline_baselines_first_model.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 02_full_results_noise_ablations.ipynb
cd reports && python build_report1.py && python build_report2.py
```

Notebook 1 takes about a minute. With the cached tables in `results/` present, notebook 2 finishes in a few minutes; without them, or with `FORCE = True` in its first code cell, it retrains everything (roughly 1.5 hours on two CPU cores). Interrupted grids resume from the last finished chunk.

## Method in brief

* **Folds** — stratified 5-fold, `random_state=42`, saved once and reused everywhere. Seeds change only the initial circuit parameters.
* **Preprocessing** — fitted on the training fold only: standardise, PCA to the number of features the circuit loads, then min-max to `[0, π]` (angle, IQP/ZZ) or to `[0, 1]` followed by L2 normalisation (amplitude).
* **Encodings** — angle: `RY(x_i)` per qubit; IQP/ZZ: the feature map of Havlíček et al. (2019), two repetitions, all pairs; amplitude: the normalised vector as the state, compiled with the Möttönen state preparation for gate counts and noise.
* **Ansatz** — `L` layers of `RZ·RY·RZ` on every qubit plus a CNOT ring, identical for all encodings. Classes are read out as basis states of the first one or two qubits; there is no classical layer after the circuit.
* **Training** — exact state-vector simulation in PennyLane, cross-entropy loss, Adam (lr 0.05, 150 full-batch epochs).
* **Noise model** — density-matrix simulation with depolarising noise after every gate (1e-3 one-qubit, 1e-2 on both qubits of a CNOT), 2e-2 read-out bit flips and 1024 shots; all rates are scaled together.
* **Ablations** — ansatz depth, qubit budget, ZZ repetitions and coupling, input scaling, ansatz entanglement, amplitude preprocessing.

## References

* M. Schuld and F. Petruccione, *Machine Learning with Quantum Computers*, Springer (2021).
* V. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces", *Nature* 567, 209 (2019).
