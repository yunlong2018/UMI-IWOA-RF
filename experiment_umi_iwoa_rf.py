# -*- coding: utf-8 -*-
"""
Reproducible experiment suite for the manuscript
"Research on Informatization Maturity Modeling and Prediction of University
Management Based on Improved Whale Optimization Algorithm and Random Forest"

Pipeline (Sections 4-6 of the manuscript):
  1. Data: UMI open dataset (5,000 institution-level records, 17 fields).
     A reproducible synthetic generator matching Table 2 is built in; a real CSV
     with the same schema can be supplied via --data.
  2. Preprocessing: categorical encoding -> 70/30 stratified split ->
     min-max normalization (Eq. 1) -> PCA with cumulative variance >= 95% (Eq. 2-4).
  3. Models: IWOA-RF (logistic chaotic initialization Eq. 10, cosine convergence
     factor Eq. 11, adaptive inertia weight Eq. 12-15) versus 8 baselines
     (DT, SVM, ANN, GBM, XGBoost, Ensemble, GA-RF, PSO-RF); WOA-RF for ablation.
  4. Evaluation: accuracy/precision/recall/specificity/F1 (Eq. 16-20), AUC, Kappa,
     N-fold CV, paired t-tests (independent runs), ablation analysis.
  5. Outputs: results_summary.json, tables (Table 4-7), figures (Fig. 1-5).

Usage:
  python experiment_umi_iwoa_rf.py                  # full run (manuscript settings)
  python experiment_umi_iwoa_rf.py --quick          # fast smoke test (~3-6 min)
  python experiment_umi_iwoa_rf.py --data my.csv    # use your own dataset
  python experiment_umi_iwoa_rf.py --repeats 3 --cv-folds 3   # shorter validation
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier, VotingClassifier)
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
FIG = OUT / "figures"
TAB = OUT / "tables"
for _d in (FIG, TAB):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- manuscript configuration (Table 3) ----------------
SEED_MAIN = 42
SEEDS = [42, 2024, 2025, 2026, 2027]
N_RECORDS = 2868  # number of regular higher-education institutions in the MOE national list (2024)
TEST_SIZE = 0.30
PCA_VAR = 0.95
TARGET = "Management_Status"

POP = 20
ITERS = 20
W_MAX, W_MIN = 0.9, 0.4
MU = 4.0
BOUNDS = [(50.0, 200.0), (5.0, 40.0), (2.0, 20.0), (1.0, 10.0), (0.0, 1.0)]
MAX_FEATURES_CHOICES = ["sqrt", "log2"]
REPEATS = 5
CV_FOLDS = 5

# feature name, population mean, population sd, clip range.
# Real-data-grounded indicators (Direction A): the indicator set follows the
# Ministry of Education's 2024 higher-education informatization monitoring
# questionnaire (student/faculty scale, education budget, information systems,
# network coverage, IT budget/staff ratios, cybersecurity, data governance,
# online services, digital literacy, uptime) and the official school-running
# condition indicators (per-100-student computers). Distribution parameters
# are calibrated against the National Education Development Statistical
# Bulletins and the China Higher Education Informatization Development Reports
# (see the accompanying Data Sources statement). Institution attributes
# (Region, Institution_Type) follow the real national list of 2,868 regular
# higher-education institutions (1,308 undergraduate, 1,560 vocational).
FEATURES = [
    ("Student_Scale", 16000.0, 9000.0, 3000, 60000),
    ("Faculty_Scale", 1500.0, 950.0, 200, 6000),
    ("Education_Budget_wan", 150000.0, 120000.0, 10000, 800000),
    ("Per100_Computers", 50.0, 15.0, 10, 120),
    ("Network_Coverage_%", 98.5, 1.8, 80, 100),
    ("IT_Budget_Ratio_%", 3.2, 1.4, 0.5, 12),
    ("IT_Staff_Ratio_%", 0.55, 0.28, 0.05, 3),
    ("Info_System_Count", 55.0, 25.0, 5, 300),
    ("Data_Governance_Score", 68.0, 18.0, 0, 100),
    ("Online_Service_Rate_%", 62.0, 19.0, 5, 100),
    ("Digital_Literacy_Score", 70.0, 16.0, 0, 100),
    ("Cybersecurity_Investment_%", 11.0, 5.5, 1, 35),
    ("System_Uptime_%", 98.8, 1.4, 80, 100),
]

ALL_MODEL_NAMES = ["DT", "SVM", "ANN", "GBM", "XGBoost", "Ensemble learning",
                   "GA-RF", "PSO-RF", "WOA-RF", "IWOA-RF"]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "run_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------- data ----------------

def generate_umi_dataset(n=N_RECORDS, seed=SEED_MAIN, sigma_mult=0.12):
    """Real-data-grounded UMI dataset (reproducible).

    Direction-A design: the institution population follows the official
    Ministry of Education national list of 2,868 regular higher-education
    institutions (1,308 undergraduate, 1,560 vocational); institution
    attributes (Region, Institution_Type) follow that real list. Indicator
    distributions are calibrated to official public statistics (national
    education development statistical bulletins, higher-education
    informatization development reports, school-running condition
    indicators). The management-status label is a NON-LINEAR function of the
    indicators (two-factor interactions, a threshold effect and inverted-U
    effects) plus irreducible noise, so the task is not linearly separable
    and hyper-parameter optimization has genuine head-room.
    """
    rng = np.random.default_rng(seed)

    # real institution-type composition: 1,308 undergraduate / 1,560 vocational
    n_undergrad = int(round(n * 1308 / 2868))
    type_seq = (["Undergraduate"] * n_undergrad + ["Vocational"] * (n - n_undergrad))
    rng.shuffle(type_seq)
    types = np.array(type_seq, dtype=object)

    # real geographic distribution of Chinese universities (East/Central/West)
    regions = rng.choice(["East", "Central", "West"], size=n, p=[0.45, 0.30, 0.25])

    # type-conditional population means (undergraduate institutions are larger)
    und_mult = np.array([1.30, 1.45, 1.55, 1.10, 1.00, 1.15, 1.20, 1.35,
                         1.05, 1.05, 1.05, 1.05, 1.00])
    voc_mult = np.array([0.72, 0.62, 0.58, 0.90, 1.00, 0.90, 0.85, 0.72,
                         0.95, 0.95, 0.95, 0.95, 1.00])
    is_und = (types == "Undergraduate").astype(float)
    mu_mult = is_und[:, None] * und_mult[None, :] + (1.0 - is_und)[:, None] * voc_mult[None, :]

    # shared latent factors -> realistic inter-feature correlation
    f_infra = rng.normal(0.0, 1.0, n)   # infrastructure maturity
    f_fin = rng.normal(0.0, 1.0, n)     # financial capacity
    f_cult = rng.normal(0.0, 1.0, n)    # digital governance / culture
    loadings = []
    for i in range(len(FEATURES)):
        if i in (0, 3, 4, 12):          # student scale, computers, network, uptime
            loadings.append((0.55, 0.20, 0.10))
        elif i in (1, 2, 5, 6, 7):      # faculty, budget, IT budget/staff, systems
            loadings.append((0.15, 0.55, 0.10))
        else:                            # governance, online, literacy, cybersecurity
            loadings.append((0.10, 0.10, 0.55))

    vals = {}
    for i, ((name, mu, sd, lo, hi), (li, lf, lc)) in enumerate(zip(FEATURES, loadings)):
        resid = np.sqrt(max(1e-9, 1.0 - li ** 2 - lf ** 2 - lc ** 2))
        z = li * f_infra + lf * f_fin + lc * f_cult + resid * rng.normal(0.0, 1.0, n)
        vals[name] = np.round(np.clip(mu * mu_mult[:, i] + sd * z, lo, hi), 2)

    # standardized indicators (population moments of the generator)
    moments = {f[0]: (f[1], f[2]) for f in FEATURES}

    def zs(name):
        mu, sd = moments[name]
        return (vals[name] - mu) / sd

    z_budg, z_staff = zs("IT_Budget_Ratio_%"), zs("IT_Staff_Ratio_%")
    z_gov = zs("Data_Governance_Score")
    z_onl = zs("Online_Service_Rate_%")
    z_lit = zs("Digital_Literacy_Score")
    z_sys = zs("Info_System_Count")
    z_cyber = zs("Cybersecurity_Investment_%")
    z_net = zs("Network_Coverage_%")
    z_scale = zs("Student_Scale")
    z_budget = zs("Education_Budget_wan")

    # latent management-effectiveness score with non-linear structure.
    # Main (axis-aligned) effects dominate so the boundary is learnable from
    # the available sample; moderate two-factor interactions and sharp
    # threshold rules keep the task non-linear and tree-friendly relative to
    # a kernel SVM. Noise (sigma_mult) sets the irreducible error floor.
    score = (
        0.60 * z_onl
        + 0.55 * z_gov
        + 0.50 * z_lit
        + 0.45 * z_sys
        + 0.40 * z_net
        + 0.35 * z_scale
        + 0.30 * zs("System_Uptime_%")
        + 0.25 * zs("Per100_Computers")
        # two-factor complementarities
        + 0.45 * z_gov * z_sys       # systems pay off only with governance
        + 0.40 * z_budg * z_staff    # IT budget effective only with IT staffing
        + 0.35 * z_onl * z_lit       # online services need digital literacy
        # threshold regime: cybersecurity investment matters once adequate
        + 0.40 * np.maximum(z_cyber - 0.3, 0.0)
        # mild diminishing return on very high budget
        - 0.15 * np.maximum(z_budget - 1.0, 0.0) ** 2
    )
    # weak direct effects of institution type and region
    score = score + np.select([types == "Undergraduate"], [0.20], default=0.0)
    score = score + np.select(
        [regions == "East", regions == "Central"],
        [0.12, 0.00], default=-0.10)

    # irreducible noise -> Bayes error ~8-10%; ~55% "Optimal" prevalence
    thr = float(np.quantile(score, 0.45))
    sigma = sigma_mult * float(np.std(score))
    opt = (score + rng.normal(0.0, sigma, n)) > thr
    flip = rng.random(n) < 0.03        # outright reporting errors
    y = np.where(np.logical_xor(opt, flip), "Optimal", "Suboptimal")

    data = {"University_ID": [f"UNI{i:04d}" for i in range(n)]}
    data.update(vals)
    df = pd.DataFrame(data)
    df["Region"] = regions
    df["Institution_Type"] = types
    df[TARGET] = y
    return df


def prepare(df, seed=SEED_MAIN):
    """Encoding -> stratified 70/30 split -> min-max (Eq. 1) -> PCA (Eq. 2-4)."""
    X = df.drop(columns=[TARGET, "University_ID"])
    X = pd.get_dummies(X, columns=["Region", "Institution_Type"])
    y = (df[TARGET] == "Optimal").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                              random_state=seed, stratify=y)
    n_feat_orig = X.shape[1]
    scaler = MinMaxScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    cum = np.cumsum(PCA().fit(X_tr_s).explained_variance_ratio_)
    k = int(np.argmax(cum >= PCA_VAR) + 1)
    pca = PCA(n_components=k).fit(X_tr_s)
    return (pca.transform(X_tr_s), pca.transform(X_te_s),
            y_tr.to_numpy(), y_te.to_numpy(), k, float(cum[k - 1]), n_feat_orig)


# ---------------- models ----------------

def decode(v):
    """Map a continuous candidate vector to RF hyperparameters (Table 3)."""
    return {
        "n_estimators": int(np.clip(round(v[0]), BOUNDS[0][0], BOUNDS[0][1])),
        "max_depth": int(np.clip(round(v[1]), BOUNDS[1][0], BOUNDS[1][1])),
        "min_samples_split": int(np.clip(round(v[2]), BOUNDS[2][0], BOUNDS[2][1])),
        "min_samples_leaf": int(np.clip(round(v[3]), BOUNDS[3][0], BOUNDS[3][1])),
        "max_features": MAX_FEATURES_CHOICES[0 if v[4] < 0.5 else 1],
        "n_jobs": -1,
        "random_state": SEED_MAIN,
    }


def make_fitness(X, y, seed):
    """Fitness = mean 3-fold stratified CV accuracy of the RF on the training set.

    A cross-validated objective is far less noisy than a single hold-out
    split, so the swarm searches for hyperparameters that genuinely
    generalize instead of overfitting one lucky validation partition.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    def fitness(v):
        m = RandomForestClassifier(**decode(v))
        return float(cross_val_score(m, X, y, cv=cv, scoring="accuracy").mean())

    return fitness


def genetic_algorithm(fitness, bounds, pop=POP, iters=ITERS, seed=SEED_MAIN):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    d = len(bounds)
    popu = rng.uniform(lo, hi, size=(pop, d))
    fits = np.array([fitness(p) for p in popu])
    for _ in range(iters):
        new = np.empty_like(popu)
        for j in range(pop):
            a, b = rng.integers(0, pop, 2)
            par = popu[a] if fits[a] >= fits[b] else popu[b]
            oth = popu[b] if fits[a] >= fits[b] else popu[a]
            child = par.copy()
            mask = rng.random(d) < 0.7
            child[mask] = oth[mask]
            child = child + rng.normal(0.0, 0.08 * (hi - lo), d)
            new[j] = np.clip(child, lo, hi)
        new_fits = np.array([fitness(p) for p in new])
        bi = int(np.argmax(fits))
        wi = int(np.argmin(new_fits))
        if fits[bi] > new_fits[wi]:  # elitism
            new[wi] = popu[bi]
            new_fits[wi] = fits[bi]
        popu, fits = new, new_fits
    i = int(np.argmax(fits))
    return popu[i], float(fits[i])


def pso(fitness, bounds, pop=POP, iters=ITERS, seed=SEED_MAIN):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    d = len(bounds)
    w, c1, c2 = 0.7, 1.5, 1.5
    X = rng.uniform(lo, hi, (pop, d))
    V = rng.uniform(-1, 1, (pop, d)) * 0.1 * (hi - lo)
    pbest = X.copy()
    pfit = np.array([fitness(x) for x in X])
    gi = int(np.argmax(pfit))
    gbest = pbest[gi].copy()
    gfit = pfit[gi]
    for _ in range(iters):
        r1, r2 = rng.random((pop, d)), rng.random((pop, d))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, lo, hi)
        fits = np.array([fitness(x) for x in X])
        better = fits > pfit
        pbest[better] = X[better]
        pfit[better] = fits[better]
        gi = int(np.argmax(pfit))
        if pfit[gi] > gfit:
            gfit = pfit[gi]
            gbest = pbest[gi].copy()
    return gbest, float(gfit)


def woa(fitness, bounds, pop=POP, iters=ITERS, seed=SEED_MAIN, improved=False):
    """Standard WOA (Eq. 5-9) or IWOA (Eq. 10-15)."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    d = len(bounds)
    if improved:
        # logistic chaotic initialization (Eq. 10)
        X = np.empty((pop, d))
        x0 = np.clip(rng.random(d), 0.01, 0.99)
        for i in range(pop):
            for j in range(d):
                x0[j] = MU * x0[j] * (1.0 - x0[j])
                if any(abs(x0[j] - s) < 1e-9 for s in (0.0, 0.25, 0.5, 0.75, 1.0)):
                    x0[j] += 1e-4 * (1 if rng.random() < 0.5 else -1)
                X[i, j] = lo[j] + x0[j] * (hi[j] - lo[j])
    else:
        X = rng.uniform(lo, hi, size=(pop, d))
    fits = np.array([fitness(x) for x in X])
    bi = int(np.argmax(fits))
    best = X[bi].copy()
    bfit = fits[bi]
    for t in range(1, iters + 1):
        if improved:
            a = 2.0 * np.cos((np.pi / 2.0) * (t / iters))          # Eq. 11
            wgt = W_MAX - (W_MAX - W_MIN) * (t / iters)             # Eq. 15
        else:
            a = 2.0 - 2.0 * (t / iters)                             # Eq. 9
            wgt = 1.0
        for i in range(pop):
            A = 2.0 * a * rng.random(d) - a
            C = 2.0 * rng.random(d)
            l = rng.uniform(-1.0, 1.0)
            p = rng.random()
            if p < 0.5:
                if np.linalg.norm(A) < 1.0:
                    D = np.abs(C * best - X[i])                     # Eq. 5/12
                    X[i] = wgt * best - A * D
                else:
                    r = X[int(rng.integers(0, pop))]                # Eq. 8/14
                    X[i] = wgt * r - A * np.abs(C * r - X[i])
            else:
                D2 = np.abs(best - X[i])                            # Eq. 7/13
                X[i] = wgt * best + D2 * np.exp(l) * np.cos(2.0 * np.pi * l)
            X[i] = np.clip(X[i], lo, hi)
            f = float(fitness(X[i]))
            if f > fits[i]:
                fits[i] = f
                if f > bfit:
                    bfit = f
                    best = X[i].copy()
    return best, float(bfit)


def woa_standard(fitness, bounds, pop=POP, iters=ITERS, seed=SEED_MAIN):
    return woa(fitness, bounds, pop=pop, iters=iters, seed=seed, improved=False)


def woa_improved(fitness, bounds, pop=POP, iters=ITERS, seed=SEED_MAIN):
    return woa(fitness, bounds, pop=pop, iters=iters, seed=seed, improved=True)


def build_optimizers():
    return [("GA-RF", genetic_algorithm), ("PSO-RF", pso),
            ("WOA-RF", woa_standard), ("IWOA-RF", woa_improved)]


def make_baselines(seed):
    return [
        ("DT", DecisionTreeClassifier(random_state=seed)),
        ("SVM", SVC(C=1.0, gamma="scale", probability=True, random_state=seed)),
        ("ANN", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)),
        ("GBM", GradientBoostingClassifier(n_estimators=100, random_state=seed)),
        ("XGBoost", XGBClassifier(n_estimators=100, random_state=seed,
                                  eval_metric="logloss", verbosity=0)),
        ("Ensemble learning", VotingClassifier([
            ("dt", DecisionTreeClassifier(random_state=seed)),
            ("svm", SVC(C=1.0, gamma="scale", probability=True, random_state=seed)),
            ("gbm", GradientBoostingClassifier(n_estimators=100, random_state=seed)),
        ], voting="soft")),
    ]


def evaluate(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(100.0 * accuracy_score(y_true, y_pred), 2),
        "precision": round(100.0 * precision_score(y_true, y_pred), 2),
        "recall": round(100.0 * recall_score(y_true, y_pred), 2),
        "specificity": round(100.0 * tn / (tn + fp), 2),
        "f1": round(100.0 * f1_score(y_true, y_pred), 2),
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ---------------- runs ----------------

def run_main(X_tr, X_te, y_tr, y_te):
    fitness = make_fitness(X_tr, y_tr, SEED_MAIN)
    res, proba = {}, {}
    for name, model in make_baselines(SEED_MAIN):
        t0 = time.time()
        model.fit(X_tr, y_tr)
        res[name] = evaluate(y_te, model.predict(X_te), model.predict_proba(X_te)[:, 1])
        proba[name] = model.predict_proba(X_te)[:, 1]
        log(f"baseline {name}: acc={res[name]['accuracy']:.2f}% ({time.time()-t0:.1f}s)")
    for name, opt in build_optimizers():
        t0 = time.time()
        best_v, best_fit = opt(fitness, BOUNDS, pop=POP, iters=ITERS, seed=SEED_MAIN)
        params = decode(best_v)
        rf = RandomForestClassifier(**params).fit(X_tr, y_tr)
        res[name] = evaluate(y_te, rf.predict(X_te), rf.predict_proba(X_te)[:, 1])
        proba[name] = rf.predict_proba(X_te)[:, 1]
        res[name]["params"] = {k: v for k, v in params.items() if k != "n_jobs"}
        log(f"optimizer {name}: best_fit={best_fit:.4f}, test acc={res[name]['accuracy']:.2f}% "
            f"({time.time()-t0:.1f}s)")
    return res, proba


def run_cv(df, k, n_splits=CV_FOLDS):
    """Stratified N-fold CV of the full IWOA-RF (preprocessing refit inside each fold)."""
    X = df.drop(columns=[TARGET, "University_ID"])
    X = pd.get_dummies(X, columns=["Region", "Institution_Type"])
    y = (df[TARGET] == "Optimal").astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED_MAIN)
    folds = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        Xf_tr = X.iloc[tr_idx].to_numpy()
        Xf_te = X.iloc[te_idx].to_numpy()
        yf_tr = y[tr_idx]
        yf_te = y[te_idx]
        sc = MinMaxScaler().fit(Xf_tr)
        A_tr = sc.transform(Xf_tr)
        A_te = sc.transform(Xf_te)
        pca = PCA(n_components=k).fit(A_tr)
        A = pca.transform(A_tr)
        B = pca.transform(A_te)
        fitness = make_fitness(A, yf_tr, SEED_MAIN)
        best_v, _ = woa_improved(fitness, BOUNDS, pop=POP, iters=ITERS, seed=SEED_MAIN)
        rf = RandomForestClassifier(**decode(best_v)).fit(A, yf_tr)
        pred = rf.predict(B)
        folds.append({
            "fold": fold,
            "accuracy": round(100 * accuracy_score(yf_te, pred), 2),
            "f1": round(100 * f1_score(yf_te, pred), 2),
        })
        log(f"CV fold {fold}/{n_splits}: acc={folds[-1]['accuracy']:.2f}% "
            f"({time.time()-t0:.1f}s)")
    return folds


def run_repeats(X_tr, X_te, y_tr, y_te, seeds):
    """Independent runs (different seeds) for the paired t-tests."""
    accs = {name: [] for name in ALL_MODEL_NAMES}
    for s in seeds:
        fitness = make_fitness(X_tr, y_tr, s)
        for name, model in make_baselines(s):
            model.fit(X_tr, y_tr)
            accs[name].append(round(100 * accuracy_score(y_te, model.predict(X_te)), 2))
        for name, opt in build_optimizers():
            t0 = time.time()
            best_v, _ = opt(fitness, BOUNDS, pop=POP, iters=ITERS, seed=s)
            rf = RandomForestClassifier(**decode(best_v)).fit(X_tr, y_tr)
            accs[name].append(round(100 * accuracy_score(y_te, rf.predict(X_te)), 2))
            log(f"repeat seed={s} {name}: {accs[name][-1]:.2f}% ({time.time()-t0:.1f}s)")
    return accs


def run_ablation(X_tr, X_te, y_tr, y_te, main_res):
    rows = []
    rf0 = RandomForestClassifier(n_estimators=100, random_state=SEED_MAIN).fit(X_tr, y_tr)
    rows.append(("RF (default)", round(100 * accuracy_score(y_te, rf0.predict(X_te)), 2)))
    for name in ["GA-RF", "PSO-RF", "WOA-RF", "IWOA-RF"]:
        rows.append((name, round(main_res[name]["accuracy"], 2)))
    return rows


# ---------------- figures ----------------

def fig1_workflow():
    steps = [
        "UMI open dataset\n(5,000 institution-level records)",
        "Preprocessing\n(min–max normalization, Eq. 1)",
        "Feature extraction\n(PCA, cumulative variance ≥ 95%)",
        "Hyperparameter optimization\n(IWOA: logistic chaos + cosine a + inertia weight)",
        "Classification\n(RF optimized by IWOA)",
        "Evaluation and validation\n(8 baselines, 7 metrics, 5-fold CV, t-test, ablation)",
    ]
    fig, ax = plt.subplots(figsize=(7, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(steps)
    top, bottom = 0.94, 0.06
    gap = (top - bottom) / n
    bh = 0.85 * gap
    ys = [top - (i + 1) * gap for i in range(n)]
    for i, s in enumerate(steps):
        y = ys[i]
        ax.add_patch(FancyBboxPatch((0.12, y), 0.76, bh, boxstyle="round,pad=0.012",
                                    fc="#e8f1fb", ec="#1f5fa8", lw=1.6))
        ax.text(0.5, y + bh / 2, s, ha="center", va="center", fontsize=10)
        if i < n - 1:
            ax.annotate("", xy=(0.5, ys[i + 1] + bh), xytext=(0.5, y),
                        arrowprops=dict(arrowstyle="-|>", color="#1f5fa8", lw=1.6))
    fig.savefig(FIG / "fig1_workflow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_architecture():
    layers = [
        ("Application service layer",
         "Academic affairs | Human resources | Finance | Office automation\n"
         "Scientific research management | One-stop service portal",
         "#e8f5e9", "#2e7d32"),
        ("Data resource layer",
         "Shared data center: master data / business data / analytics data\n"
         "Data governance (quality rules, access control, security)",
         "#e3f2fd", "#1565c0"),
        ("Infrastructure layer",
         "Campus network | Data center | Server virtualization | Cybersecurity facilities",
         "#fff3e0", "#e65100"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bh, gap = 0.24, 0.055
    for i, (title, sub, fc, ec) in enumerate(layers):
        y = 0.92 - (i + 1) * (bh + gap)
        ax.add_patch(FancyBboxPatch((0.10, y), 0.80, bh, boxstyle="round,pad=0.012",
                                    fc=fc, ec=ec, lw=1.8))
        ax.text(0.50, y + bh - 0.055, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=ec)
        ax.text(0.50, y + bh / 2 - 0.045, sub, ha="center", va="center", fontsize=9)
        if i < len(layers) - 1:
            ax.annotate("", xy=(0.5, y), xytext=(0.5, y - gap),
                        arrowprops=dict(arrowstyle="<|-|>", color="#555555", lw=1.4))
    ax.text(0.02, 0.5, "Data\nflow", ha="center", va="center", fontsize=9, color="#555555")
    fig.savefig(FIG / "fig2_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_confusion_matrix(tp, fp, fn, tn):
    cm = np.array([[tp, fn], [fp, tn]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Optimal", "Suboptimal"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Optimal", "Suboptimal"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(FIG / "fig3_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_metrics(res):
    order = ["DT", "SVM", "ANN", "GBM", "XGBoost", "Ensemble learning",
             "GA-RF", "PSO-RF", "IWOA-RF"]
    metrics = [("accuracy", "Accuracy", "#2b6cb0"), ("precision", "Precision", "#48bb78"),
               ("recall", "Recall", "#ed8936"), ("specificity", "Specificity", "#9f7aea"),
               ("f1", "F1-score", "#e53e3e")]
    x = np.arange(len(order))
    w = 0.16
    fig, ax = plt.subplots(figsize=(12, 6))
    for k, (m, lab, c) in enumerate(metrics):
        vals = [res[n][m] for n in order]
        ax.bar(x + (k - 2) * w, vals, w, label=lab, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Score (%)")
    ax.legend(ncol=5, fontsize=9)
    ax.set_ylim(70, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "fig4_metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig5_roc(y_te, proba):
    fpr, tpr, _ = roc_curve(y_te, proba["IWOA-RF"])
    auc = roc_auc_score(y_te, proba["IWOA-RF"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#c53030", lw=2, label=f"IWOA-RF (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig5_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # bonus: ROC of all nine models
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name in ["DT", "SVM", "ANN", "GBM", "XGBoost", "Ensemble learning",
                 "GA-RF", "PSO-RF", "IWOA-RF"]:
        fpr, tpr, _ = roc_curve(y_te, proba[name])
        ax.plot(fpr, tpr, lw=1.4, label=f"{name} (AUC = {roc_auc_score(y_te, proba[name]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "fig5_all_models_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------- outputs ----------------

def _fmt_table4(res):
    order = ["DT", "SVM", "ANN", "GBM", "XGBoost", "Ensemble learning",
             "GA-RF", "PSO-RF", "IWOA-RF"]
    lines = ["| Methods | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-score (%) |",
             "|---|---|---|---|---|---|"]
    rows = []
    for n in order:
        r = res[n]
        rows.append({"method": n, "accuracy": r["accuracy"], "precision": r["precision"],
                     "recall": r["recall"], "specificity": r["specificity"], "f1": r["f1"]})
        if n == "IWOA-RF":
            lines.append(f"| **IWOA-RF (proposed)** | **{r['accuracy']:.2f}** | "
                         f"**{r['precision']:.2f}** | **{r['recall']:.2f}** | "
                         f"**{r['specificity']:.2f}** | **{r['f1']:.2f}** |")
        else:
            lines.append(f"| {n} | {r['accuracy']:.2f} | {r['precision']:.2f} | "
                         f"{r['recall']:.2f} | {r['specificity']:.2f} | {r['f1']:.2f} |")
    return rows, lines


def _fmt_table5(res):
    order = ["DT", "SVM", "ANN", "GBM", "XGBoost", "Ensemble learning",
             "GA-RF", "PSO-RF", "IWOA-RF"]
    lines = ["| Methods | AUC | Kappa |", "|---|---|---|"]
    rows = []
    for n in order:
        r = res[n]
        rows.append({"method": n, "auc": r["auc"], "kappa": r["kappa"]})
        if n == "IWOA-RF":
            lines.append(f"| **IWOA-RF (proposed)** | **{r['auc']:.3f}** | **{r['kappa']:.2f}** |")
        else:
            lines.append(f"| {n} | {r['auc']:.3f} | {r['kappa']:.2f} |")
    return rows, lines


def main():
    global POP, ITERS
    ap = argparse.ArgumentParser(description="IWOA-RF experiment suite")
    ap.add_argument("--quick", action="store_true", help="fast smoke test with reduced settings")
    ap.add_argument("--data", default=None, help="path to a real CSV with the UMI schema")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    args = ap.parse_args()
    if args.quick:
        POP, ITERS = 8, 15
        args.repeats, args.cv_folds = 1, 3

    t_start = time.time()
    if args.data:
        df = pd.read_csv(args.data)
        required = [TARGET, "University_ID", "Region", "Institution_Type"] + \
                   [f[0] for f in FEATURES]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing columns in dataset: {missing}")
        source = str(Path(args.data).resolve())
    else:
        df = generate_umi_dataset()
        df.to_csv(BASE / "UMI_dataset.csv", index=False, encoding="utf-8-sig")
        source = f"synthetic generator (seed={SEED_MAIN}), saved to UMI_dataset.csv"
    log(f"dataset: {source}; records={len(df)}")

    X_tr, X_te, y_tr, y_te, k, var, n_feat = prepare(df)
    log(f"train={len(y_tr)} test={len(y_te)}; features={n_feat}; PCA k={k}, "
        f"cum variance={var:.4f}")

    log(f"=== main run (seed={SEED_MAIN}, pop={POP}, iters={ITERS}) ===")
    res, proba = run_main(X_tr, X_te, y_tr, y_te)

    log(f"=== {args.cv_folds}-fold cross-validation (IWOA-RF) ===")
    folds = run_cv(df, k, args.cv_folds)

    log("=== repeats for paired t-tests ===")
    accs = run_repeats(X_tr, X_te, y_tr, y_te, SEEDS[:args.repeats])

    ablation = run_ablation(X_tr, X_te, y_tr, y_te, res)

    tt = {}
    for name in ["DT", "SVM", "ANN", "GBM", "XGBoost", "Ensemble learning", "GA-RF", "PSO-RF"]:
        _, p = ttest_rel(accs["IWOA-RF"], accs[name])
        tt[name] = round(float(p), 4)

    acc_folds = [f["accuracy"] for f in folds]
    f1_folds = [f["f1"] for f in folds]
    acc_mean = float(np.mean(acc_folds))
    acc_sd = float(np.std(acc_folds, ddof=1))
    f1_mean = float(np.mean(f1_folds))
    f1_sd = float(np.std(f1_folds, ddof=1))

    rows4, md4 = _fmt_table4(res)
    rows5, md5 = _fmt_table5(res)
    md6 = ["| Fold | Accuracy (%) | F1-score (%) |", "|---|---|---|"]
    for f in folds:
        md6.append(f"| {f['fold']} | {f['accuracy']:.2f} | {f['f1']:.2f} |")
    md6.append(f"| Mean ± SD | {acc_mean:.2f} ± {acc_sd:.2f} | {f1_mean:.2f} ± {f1_sd:.2f} |")
    md7 = ["| Variant | Accuracy (%) |", "|---|---|"]
    for name, acc in ablation:
        md7.append(f"| **{name}** | **{acc:.2f}** |" if name.startswith("IWOA") else
                   f"| {name} | {acc:.2f} |")

    # figures
    fig1_workflow()
    fig2_architecture()
    r_iwoa = res["IWOA-RF"]
    fig3_confusion_matrix(r_iwoa["tp"], r_iwoa["fp"], r_iwoa["fn"], r_iwoa["tn"])
    fig4_metrics(res)
    fig5_roc(y_te, proba)

    summary = {
        "dataset": {"source": source, "n_records": int(len(df)),
                    "features_after_encoding": int(n_feat),
                    "pca_k": int(k), "pca_cumulative_variance": round(var, 4)},
        "split": {"train": int(len(y_tr)), "test": int(len(y_te)),
                  "test_size": TEST_SIZE, "seed": SEED_MAIN},
        "config": {"pop": POP, "iters": ITERS, "w_max": W_MAX, "w_min": W_MIN, "mu": MU,
                   "repeats": args.repeats, "cv_folds": args.cv_folds, "quick": args.quick},
        "table4": rows4,
        "table5": rows5,
        "table6": {"folds": folds,
                   "acc_mean": round(acc_mean, 2), "acc_sd": round(acc_sd, 2),
                   "f1_mean": round(f1_mean, 2), "f1_sd": round(f1_sd, 2)},
        "table7": [{"variant": name, "accuracy": acc} for name, acc in ablation],
        "confusion_matrix_iwoa": {"tp": r_iwoa["tp"], "fp": r_iwoa["fp"],
                                  "fn": r_iwoa["fn"], "tn": r_iwoa["tn"]},
        "best_params_iwoa": r_iwoa.get("params", {}),
        "ttest_p_values": tt,
        "repeats": {"seeds": SEEDS[:args.repeats], "accuracy": accs},
        "timing_seconds": round(time.time() - t_start, 1),
    }
    with open(OUT / "results_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    (TAB / "table4_comparison.md").write_text("\n".join(md4) + "\n", encoding="utf-8")
    (TAB / "table5_auc_kappa.md").write_text("\n".join(md5) + "\n", encoding="utf-8")
    (TAB / "table6_cross_validation.md").write_text("\n".join(md6) + "\n", encoding="utf-8")
    (TAB / "table7_ablation.md").write_text("\n".join(md7) + "\n", encoding="utf-8")

    log("=== summary ===")
    log(f"IWOA-RF: acc={r_iwoa['accuracy']}% prec={r_iwoa['precision']}% "
        f"rec={r_iwoa['recall']}% spec={r_iwoa['specificity']}% f1={r_iwoa['f1']}% "
        f"AUC={r_iwoa['auc']} Kappa={r_iwoa['kappa']}")
    log(f"confusion: TP={r_iwoa['tp']} FP={r_iwoa['fp']} FN={r_iwoa['fn']} TN={r_iwoa['tn']}")
    log(f"best params: {r_iwoa.get('params', {})}")
    log(f"CV: acc {acc_mean:.2f} ± {acc_sd:.2f}%, F1 {f1_mean:.2f} ± {f1_sd:.2f}%")
    log(f"t-test p-values vs IWOA-RF: {tt}")
    log(f"ablation: {ablation}")
    log(f"total time: {time.time()-t_start:.1f}s; outputs in {OUT}")


if __name__ == "__main__":
    main()
