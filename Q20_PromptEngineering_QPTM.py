#!/usr/bin/env python3
"""
Q20 Prompt Engineering — tehnika: Quantum Prompt Template Mixture (QPTM)
(čisto kvantno: superpozicija semantički različitih „prompt template-ova" preko aux registra).

Koncept (kvantni analog klasičnog prompt engineering-a / reasoning-a):
  - U klasičnom LLM-u „prompt" usmerava model (sistem-instrukcija, few-shot,
    ugao posmatranja). Ovde imamo M = 4 SEMANTIČKI RAZLIČITA template-a,
    svaki kao amplitude-encoding svog freq_vector-a (iz CELOG CSV-a
    ili njegovog dela):
        T1 — HOT    : globalni freq_vector CELOG CSV-a.
        T2 — COLD   : inverzni freq (1/(f+ε)) CELOG CSV-a.
        T3 — RECENT : freq_vector poslednjih K redova.
        T4 — OLD    : freq_vector prvih N−K redova.
    (U T3+T4 unija je ceo CSV → pravilo 10 zadovoljeno.)

Kolo (nq + m qubit-a, m = 2 jer M = 4):
  1) H^⊗m na aux registru → uniformna superpozicija template-indeksa.
  2) Za i = 0..M-1: multi-controlled StatePreparation(|ψ_Ti⟩)
     sa ctrl_state=i nad aux, targetuje state registar.
  Rezultat: |Ψ⟩ = (1/√M) Σ_i |i⟩_aux ⊗ |ψ_Ti⟩_state.

Marginala nad state registrom:
  p[k] = (1/M) Σ_i |ψ_Ti[k]|²  →  bias_39  →  TOP-7 = NEXT.

Razlika u odnosu na Q13/Q15/Q17:
  Q13 (QCW): jedna semantička osa (vremenski window-sweep, log-spaced tail-ovi).
  Q15 (Halucinacija): binarno CSV vs inv-CSV preko aux Ry (2 template-a).
  Q17 (RLHF): binarno baseline vs preference sa Hadamard INTERFERENCIJOM + post-selekcijom.
  QPTM:       4 razna semantička template-a (hot/cold/recent/old), uniformna težina,
              marginalizacija aux-a — pravi analog PROMPT ENGINEERING-a
              (biraš kojim uglom „pitaš" model).

Sve deterministički: seed=39; sva 4 template-a izvedena iz CELOG CSV-a.
Deterministička grid-optimizacija (nq, K) po cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_K = (50, 100, 200, 500, 1000, 2000)

NUM_TEMPLATES = 4
M_AUX = 2
EPS_COLD = 1e-3


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# 4 semantička prompt-template-a
# =========================
def inverse_freq(f: np.ndarray, eps: float = EPS_COLD) -> np.ndarray:
    return 1.0 / (f + eps)


def template_amps(H: np.ndarray, nq: int, K: int) -> List[np.ndarray]:
    """T1=HOT (ceo CSV), T2=COLD (inv ceo CSV), T3=RECENT tail-K, T4=OLD (N−K)."""
    n = H.shape[0]
    K_eff = max(N_NUMBERS, min(n - N_NUMBERS, int(K)))

    f_hot = freq_vector(H)
    f_cold = inverse_freq(f_hot, EPS_COLD)
    f_rec = freq_vector(H[-K_eff:])
    f_old = freq_vector(H[: n - K_eff])

    return [
        amp_from_freq(f_hot, nq),
        amp_from_freq(f_cold, nq),
        amp_from_freq(f_rec, nq),
        amp_from_freq(f_old, nq),
    ]


# =========================
# QPTM kolo: H^⊗m + multi-ctrl StatePreparation po template-u
# =========================
def build_qptm_state(amps: List[np.ndarray], nq: int, m: int = M_AUX) -> Statevector:
    """|Ψ⟩ = (1/√M) Σ_i |i⟩_aux ⊗ |ψ_Ti⟩_state."""
    M = 2 ** m
    assert len(amps) == M, "broj template-ova mora biti 2^m"
    state = QuantumRegister(nq, name="s")
    aux = QuantumRegister(m, name="a")
    qc = QuantumCircuit(state, aux)

    qc.h(aux)

    for i in range(M):
        sp = StatePreparation(amps[i].tolist())
        sp_ctrl = sp.control(num_ctrl_qubits=m, ctrl_state=i)
        qc.append(sp_ctrl, list(aux) + list(state))

    return Statevector(qc)


def qptm_state_probs(H: np.ndarray, nq: int, K: int, m: int = M_AUX) -> np.ndarray:
    amps = template_amps(H, nq, K)
    sv = build_qptm_state(amps, nq, m)
    p = np.abs(sv.data) ** 2

    dim_s = 2 ** nq
    dim_a = 2 ** m
    mat = p.reshape(dim_a, dim_s)
    p_s = mat.sum(axis=0)
    s = float(p_s.sum())
    return p_s / s if s > 0 else p_s


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (nq, K)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        for K in GRID_K:
            try:
                p = qptm_state_probs(H, nq, int(K))
                bi = bias_39(p)
                score = cosine(bi, f_csv_n)
            except Exception:
                continue
            key = (score, nq, -int(K))
            if best is None or key > best[0]:
                best = (key, dict(nq=nq, K=int(K), score=float(score)))
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q20 Prompt Engineering (QPTM — mešavina 4 template-a): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED, "| M (template-ova):", NUM_TEMPLATES)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| K (recent/old split):", best["K"],
        "| N−K:", H.shape[0] - best["K"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    f_csv = freq_vector(H)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX

    print("--- predikcije pojedinačnih template-ova (za isti nq) ---")
    nq_best = int(best["nq"])
    K_best = int(best["K"])
    amps = template_amps(H, nq_best, K_best)
    names = ("T1 HOT   ", "T2 COLD  ", "T3 RECENT", "T4 OLD   ")
    for name, amp in zip(names, amps):
        p_i = np.abs(amp) ** 2
        pred_i = pick_next_combination(p_i)
        cos_i = cosine(bias_39(p_i), f_csv_n)
        print(f"{name}  cos={cos_i:.6f}  NEXT={pred_i}")

    p = qptm_state_probs(H, nq_best, K_best)
    pred = pick_next_combination(p)
    print("--- glavna predikcija (QPTM mešavina) ---")
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q20 Prompt Engineering (QPTM — mešavina 4 template-a): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | M (template-ova): 4
BEST hparam: nq= 5 | K (recent/old split): 2000 | N−K: 2600 | cos(bias, freq_csv): 0.901748
--- predikcije pojedinačnih template-ova (za isti nq) ---
T1 HOT     cos=0.900351  NEXT=(7, 19, 22, 24, 27, 28, 31)
T2 COLD    cos=0.899011  NEXT=(1, 13, 14, 17, 23, 25, 30)
T3 RECENT  cos=0.898246  NEXT=(7, 9, 16, 19, 24, 28, 31)
T4 OLD     cos=0.899599  NEXT=(7, 8, 10, 19, 22, 27, 32)
--- glavna predikcija (QPTM mešavina) ---
predikcija NEXT: (7, 19, 22, 24, 27, 28, 31)
"""



"""
Q20_PromptEngineering_QPTM.py — tehnika: Quantum Prompt Template Mixture (QPTM).

Koncept:
„Prompt" kao kvantna superpozicija M = 4 semantički različita template-a,
svaki je amplitude-encoding svog freq_vector-a iz CELOG CSV-a ili njegovog dela:
  T1 HOT    — globalni freq_vector CELOG CSV-a
  T2 COLD   — inverzni freq (1/(f+ε)) CELOG CSV-a
  T3 RECENT — freq_vector poslednjih K redova
  T4 OLD    — freq_vector prvih N−K redova

Kolo (nq + 2 aux qubit-a):
  H^⊗2 na aux → uniformna superpozicija 4 template-indeksa.
  Za i = 0..3: multi-controlled StatePreparation(|ψ_Ti⟩) sa ctrl_state=i.
  Rezultat: |Ψ⟩ = (1/2) Σ_{i=0..3} |i⟩_aux ⊗ |ψ_Ti⟩_state.
Marginala nad state registrom = (1/4) Σ_i |ψ_Ti|² → bias_39 → TOP-7 = NEXT.

Tehnike:
Amplitude encoding po template-u (StatePreparation).
Uniformna kvantna superpozicija preko aux indeksa (H^⊗m).
Multi-controlled state prep sa ctrl_state (ekskluzivno po template-indeksu).
Egzaktni Statevector (bez uzorkovanja).
Deterministička grid-optimizacija (nq, K).

Prednosti:
Kvantni analog prompt engineering-a: biraš 4 „ugla pitanja" i mešaš ih u jednom kolu.
4 semantički različita template-a (ne samo vremenska osa kao Q13).
Ceo CSV učestvuje (T1+T2 pokrivaju ceo CSV, T3∪T4 = ceo CSV).
Čisto kvantno: bez klasičnog treninga, bez softmax-a, bez hibrida.

Nedostaci:
Marginala je linearna mešavina template-ova (aux je ortogonalna → bez kvantne
interferencije između template-ova; to je po dizajnu, jer QPTM simulira
„paralelno pitanje sa 4 prompt-a", ne interferentnu RLHF alignment).
Multi-ctrl SP je skupo — m = 2 aux (4 template-a) je praktičan plafon sa nq ≤ 6.
mod-39 readout meša stanja (dim 2^nq ≠ 39).
"""
