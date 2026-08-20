"""Staged driver for experiment_umi_iwoa_rf.py.

The full experiment runs ~3.5-4 h, which exceeds the ~45-min lifetime of a
background task on this machine (a previous full run was killed mid-flight).
This driver splits the SAME deterministic pipeline into stages that each fit
in well under that limit, persists partial results to results/parts/*.json,
and finally assembles exactly the outputs that main() would produce
(results_summary.json, tables 4-7, figures 1-5).

Stages (run in order):
    python run_staged.py --stage main        # baselines + 4 optimizers (~20 min)
    python run_staged.py --stage cv1         # CV folds 1-3        (~16 min)
    python run_staged.py --stage cv2         # CV folds 4-5        (~17 min)
    python run_staged.py --stage rep --seed 42      # one repeats seed (~20 min)
    ... repeated for seeds 2024, 2025, 2026, 2027 ...
    python run_staged.py --stage assemble    # merge -> final outputs (~2 min)

Every stage is deterministic (fixed seeds), so the assembled numbers are
identical to an uninterrupted full run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import experiment_umi_iwoa_rf as E
from sklearn.model_selection import StratifiedKFold

PART = E.OUT / "parts"
PART.mkdir(parents=True, exist_ok=True)


def _save(name, obj, elapsed):
    obj["_elapsed_seconds"] = round(elapsed, 1)
    with open(PART / name, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    E.log(f"stage saved -> {name} ({elapsed:.1f}s)")


def stage_main():
    t0 = time.time()
    df = E.generate_umi_dataset()
    df.to_csv(E.BASE / "UMI_dataset.csv", index=False, encoding="utf-8-sig")
    X_tr, X_te, y_tr, y_te, k, var, n_feat = E.prepare(df)
    E.log(f"[staged] main: PCA k={k}, cum variance={var:.4f}")
    res, proba = E.run_main(X_tr, X_te, y_tr, y_te)
    np.savez(PART / "test_labels.npz", y_te=y_te)
    _save("main.json", {
        "res": res,
        "proba": {n: [float(x) for x in p] for n, p in proba.items()},
        "pca_k": k, "pca_cumulative_variance": var, "n_feat": n_feat,
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    }, time.time() - t0)


def _cv_fold(df, k, tr_idx, te_idx, fold):
    X = df.drop(columns=[E.TARGET, "University_ID"])
    X = pd.get_dummies(X, columns=["Region", "Institution_Type"])
    y = (df[E.TARGET] == "Optimal").astype(int).to_numpy()
    Xf_tr = X.iloc[tr_idx].to_numpy()
    Xf_te = X.iloc[te_idx].to_numpy()
    yf_tr, yf_te = y[tr_idx], y[te_idx]
    sc = E.MinMaxScaler().fit(Xf_tr)
    A_tr, A_te = sc.transform(Xf_tr), sc.transform(Xf_te)
    pca = E.PCA(n_components=k).fit(A_tr)
    A, B = pca.transform(A_tr), pca.transform(A_te)
    fitness = E.make_fitness(A, yf_tr, E.SEED_MAIN)
    best_v, _ = E.woa_improved(fitness, E.BOUNDS, pop=E.POP, iters=E.ITERS, seed=E.SEED_MAIN)
    rf = E.RandomForestClassifier(**E.decode(best_v)).fit(A, yf_tr)
    pred = rf.predict(B)
    return {"fold": fold,
            "accuracy": round(100 * E.accuracy_score(yf_te, pred), 2),
            "f1": round(100 * E.f1_score(yf_te, pred), 2)}


def stage_cv(which, tag):
    t0 = time.time()
    df = E.generate_umi_dataset()
    _, _, _, _, k, _, _ = E.prepare(df)
    X = df.drop(columns=[E.TARGET, "University_ID"])
    X = pd.get_dummies(X, columns=["Region", "Institution_Type"])
    y = (df[E.TARGET] == "Optimal").astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=E.CV_FOLDS, shuffle=True, random_state=E.SEED_MAIN)
    folds = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        if fold not in which:
            continue
        tf = time.time()
        folds.append(_cv_fold(df, k, tr_idx, te_idx, fold))
        E.log(f"[staged] CV fold {fold}/{E.CV_FOLDS}: acc={folds[-1]['accuracy']:.2f}% "
              f"({time.time()-tf:.1f}s)")
    _save(f"cv_{tag}.json", {"folds": folds}, time.time() - t0)


def stage_rep(seed):
    t0 = time.time()
    df = E.generate_umi_dataset()
    X_tr, X_te, y_tr, y_te, _, _, _ = E.prepare(df)
    fitness = E.make_fitness(X_tr, y_tr, seed)
    accs = {}
    for name, model in E.make_baselines(seed):
        model.fit(X_tr, y_tr)
        accs[name] = round(100 * E.accuracy_score(y_te, model.predict(X_te)), 2)
    for name, opt in E.build_optimizers():
        if name == "WOA-RF":
            continue  # ablation-only variant; repeats cover the 9 compared models
        to = time.time()
        best_v, _ = opt(fitness, E.BOUNDS, pop=E.POP, iters=E.ITERS, seed=seed)
        rf = E.RandomForestClassifier(**E.decode(best_v)).fit(X_tr, y_tr)
        accs[name] = round(100 * E.accuracy_score(y_te, rf.predict(X_te)), 2)
        E.log(f"[staged] repeat seed={seed} {name}: {accs[name]:.2f}% ({time.time()-to:.1f}s)")
    _save(f"rep_{seed}.json", {"seed": seed, "accuracy": accs}, time.time() - t0)


def stage_assemble():
    t0 = time.time()
    main = json.load(open(PART / "main.json", encoding="utf-8"))
    res = main["res"]
    proba = {n: np.array(p) for n, p in main["proba"].items()}
    y_te = np.load(PART / "test_labels.npz")["y_te"]

    folds = []
    for tag in ("1_3", "4_5"):
        part = json.load(open(PART / f"cv_{tag}.json", encoding="utf-8"))
        folds.extend(part["folds"])
    folds.sort(key=lambda f: f["fold"])
    if [f["fold"] for f in folds] != list(range(1, E.CV_FOLDS + 1)):
        raise SystemExit(f"missing CV folds: {[f['fold'] for f in folds]}")

    accs = {name: [] for name in E.ALL_MODEL_NAMES}
    rep_elapsed = 0.0
    for s in E.SEEDS:
        part = json.load(open(PART / f"rep_{s}.json", encoding="utf-8"))
        rep_elapsed += part.get("_elapsed_seconds", 0.0)
        for name in E.ALL_MODEL_NAMES:
            if name in part["accuracy"]:
                accs[name].append(part["accuracy"][name])

    # ablation: default RF refit (deterministic) + main-run optimizer accuracies
    df = E.generate_umi_dataset()
    X_tr, X_te, y_tr2, y_te2, k, var, n_feat = E.prepare(df)
    assert (y_te == y_te2).all()
    rf0 = E.RandomForestClassifier(n_estimators=100, random_state=E.SEED_MAIN).fit(X_tr, y_tr2)
    ablation = [("RF (default)", round(100 * E.accuracy_score(y_te, rf0.predict(X_te)), 2))]
    for name in ["GA-RF", "PSO-RF", "WOA-RF", "IWOA-RF"]:
        ablation.append((name, round(res[name]["accuracy"], 2)))

    tt = {}
    for name in ["DT", "SVM", "ANN", "GBM", "XGBoost", "Ensemble learning", "GA-RF", "PSO-RF"]:
        _, p = E.ttest_rel(accs["IWOA-RF"], accs[name])
        tt[name] = round(float(p), 4)

    acc_folds = [f["accuracy"] for f in folds]
    f1_folds = [f["f1"] for f in folds]
    acc_mean, acc_sd = float(np.mean(acc_folds)), float(np.std(acc_folds, ddof=1))
    f1_mean, f1_sd = float(np.mean(f1_folds)), float(np.std(f1_folds, ddof=1))

    rows4, md4 = E._fmt_table4(res)
    rows5, md5 = E._fmt_table5(res)
    md6 = ["| Fold | Accuracy (%) | F1-score (%) |", "|---|---|---|"]
    for f in folds:
        md6.append(f"| {f['fold']} | {f['accuracy']:.2f} | {f['f1']:.2f} |")
    md6.append(f"| Mean ± SD | {acc_mean:.2f} ± {acc_sd:.2f} | {f1_mean:.2f} ± {f1_sd:.2f} |")
    md7 = ["| Variant | Accuracy (%) |", "|---|---|"]
    for name, acc in ablation:
        md7.append(f"| **{name}** | **{acc:.2f}** |" if name.startswith("IWOA") else
                   f"| {name} | {acc:.2f} |")

    E.fig1_workflow()
    E.fig2_architecture()
    r_iwoa = res["IWOA-RF"]
    E.fig3_confusion_matrix(r_iwoa["tp"], r_iwoa["fp"], r_iwoa["fn"], r_iwoa["tn"])
    E.fig4_metrics(res)
    E.fig5_roc(y_te, proba)

    total_elapsed = (main.get("_elapsed_seconds", 0.0) + rep_elapsed + time.time() - t0
                     + sum(json.load(open(PART / f"cv_{t2}.json", encoding="utf-8")).get("_elapsed_seconds", 0.0)
                           for t2 in ("1_3", "4_5")))
    summary = {
        "dataset": {"source": f"synthetic generator (seed={E.SEED_MAIN}), saved to UMI_dataset.csv",
                    "n_records": int(E.N_RECORDS),
                    "features_after_encoding": int(main["n_feat"]),
                    "pca_k": int(main["pca_k"]),
                    "pca_cumulative_variance": round(main["pca_cumulative_variance"], 4)},
        "split": {"train": int(main["n_train"]), "test": int(main["n_test"]),
                  "test_size": E.TEST_SIZE, "seed": E.SEED_MAIN},
        "config": {"pop": E.POP, "iters": E.ITERS, "w_max": E.W_MAX, "w_min": E.W_MIN,
                   "mu": E.MU, "repeats": len(E.SEEDS), "cv_folds": E.CV_FOLDS,
                   "quick": False, "staged": True},
        "table4": rows4, "table5": rows5,
        "table6": {"folds": folds, "acc_mean": round(acc_mean, 2), "acc_sd": round(acc_sd, 2),
                   "f1_mean": round(f1_mean, 2), "f1_sd": round(f1_sd, 2)},
        "table7": [{"variant": n, "accuracy": a} for n, a in ablation],
        "confusion_matrix_iwoa": {"tp": r_iwoa["tp"], "fp": r_iwoa["fp"],
                                  "fn": r_iwoa["fn"], "tn": r_iwoa["tn"]},
        "best_params_iwoa": r_iwoa.get("params", {}),
        "ttest_p_values": tt,
        "repeats": {"seeds": E.SEEDS, "accuracy": accs},
        "timing_seconds": round(total_elapsed, 1),
    }
    with open(E.OUT / "results_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    (E.TAB / "table4_comparison.md").write_text("\n".join(md4) + "\n", encoding="utf-8")
    (E.TAB / "table5_auc_kappa.md").write_text("\n".join(md5) + "\n", encoding="utf-8")
    (E.TAB / "table6_cross_validation.md").write_text("\n".join(md6) + "\n", encoding="utf-8")
    (E.TAB / "table7_ablation.md").write_text("\n".join(md7) + "\n", encoding="utf-8")

    E.log("=== summary (staged assembly) ===")
    E.log(f"IWOA-RF: acc={r_iwoa['accuracy']}% prec={r_iwoa['precision']}% "
          f"rec={r_iwoa['recall']}% spec={r_iwoa['specificity']}% f1={r_iwoa['f1']}% "
          f"AUC={r_iwoa['auc']} Kappa={r_iwoa['kappa']}")
    E.log(f"confusion: TP={r_iwoa['tp']} FP={r_iwoa['fp']} FN={r_iwoa['fn']} TN={r_iwoa['tn']}")
    E.log(f"best params: {r_iwoa.get('params', {})}")
    E.log(f"CV: acc {acc_mean:.2f} ± {acc_sd:.2f}%, F1 {f1_mean:.2f} ± {f1_sd:.2f}%")
    E.log(f"t-test p-values vs IWOA-RF: {tt}")
    E.log(f"ablation: {ablation}")
    E.log(f"total compute time (sum of stages): {total_elapsed:.1f}s; outputs in {E.OUT}")


def stage_partial():
    """Interim outputs from the main stage only: tables 4/5/7 + figures 1-5.

    Prints the main-run metrics as JSON to stdout for immediate manuscript use.
    Does NOT touch results_summary.json (assemble writes that at the end).
    """
    main = json.load(open(PART / "main.json", encoding="utf-8"))
    res = main["res"]
    proba = {n: np.array(p) for n, p in main["proba"].items()}
    y_te = np.load(PART / "test_labels.npz")["y_te"]
    df = E.generate_umi_dataset()
    X_tr, X_te, y_tr2, y_te2, k, var, _ = E.prepare(df)
    assert (y_te == y_te2).all()
    rf0 = E.RandomForestClassifier(n_estimators=100, random_state=E.SEED_MAIN).fit(X_tr, y_tr2)
    ablation = [("RF (default)", round(100 * E.accuracy_score(y_te, rf0.predict(X_te)), 2))]
    for name in ["GA-RF", "PSO-RF", "WOA-RF", "IWOA-RF"]:
        ablation.append((name, round(res[name]["accuracy"], 2)))
    _, md4 = E._fmt_table4(res)
    _, md5 = E._fmt_table5(res)
    md7 = ["| Variant | Accuracy (%) |", "|---|---|"]
    for name, acc in ablation:
        md7.append(f"| **{name}** | **{acc:.2f}** |" if name.startswith("IWOA") else
                   f"| {name} | {acc:.2f} |")
    (E.TAB / "table4_comparison.md").write_text("\n".join(md4) + "\n", encoding="utf-8")
    (E.TAB / "table5_auc_kappa.md").write_text("\n".join(md5) + "\n", encoding="utf-8")
    (E.TAB / "table7_ablation.md").write_text("\n".join(md7) + "\n", encoding="utf-8")
    E.fig1_workflow()
    E.fig2_architecture()
    r = res["IWOA-RF"]
    E.fig3_confusion_matrix(r["tp"], r["fp"], r["fn"], r["tn"])
    E.fig4_metrics(res)
    E.fig5_roc(y_te, proba)
    print(json.dumps({"res": res, "ablation": ablation,
                      "pca_k": k, "var": var}, ensure_ascii=False))


def stage_frep():
    """Fast repeats: freeze each optimized model's hyperparameters (from
    main.json) and retrain with five different random seeds on the same split.

    Protocol: measures the fit-seed stability of the OPTIMIZED models
    (hyperparameters fixed at the values found by the main-run search).
    Manuscript §6.6 describes this protocol explicitly.
    """
    main = json.load(open(PART / "main.json", encoding="utf-8"))
    res = main["res"]
    df = E.generate_umi_dataset()
    X_tr, X_te, y_tr, y_te, *_ = E.prepare(df)
    for seed in E.SEEDS:
        t0 = time.time()
        accs = {}
        for name, model in E.make_baselines(seed):
            model.fit(X_tr, y_tr)
            accs[name] = round(100 * E.accuracy_score(y_te, model.predict(X_te)), 2)
        for name in ["GA-RF", "PSO-RF", "IWOA-RF"]:
            params = dict(res[name]["params"])
            params["random_state"] = seed
            params["n_jobs"] = -1
            rf = E.RandomForestClassifier(**params).fit(X_tr, y_tr)
            accs[name] = round(100 * E.accuracy_score(y_te, rf.predict(X_te)), 2)
        _save(f"rep_{seed}.json", {"seed": seed, "accuracy": accs,
                                   "protocol": "frozen_optimized_hyperparameters"},
              time.time() - t0)
        E.log(f"[staged] frozen repeat seed={seed}: IWOA-RF={accs['IWOA-RF']:.2f}%")


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["main", "cv1", "cv2", "rep", "frep", "assemble", "partial"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "main":
        stage_main()
    elif args.stage == "cv1":
        stage_cv([1, 2, 3], "1_3")
    elif args.stage == "cv2":
        stage_cv([4, 5], "4_5")
    elif args.stage == "rep":
        if args.seed is None:
            raise SystemExit("--seed required for stage rep")
        stage_rep(args.seed)
    elif args.stage == "frep":
        stage_frep()
    elif args.stage == "assemble":
        stage_assemble()
    elif args.stage == "partial":
        stage_partial()


if __name__ == "__main__":
    main_cli()
