#!/usr/bin/env python3
"""
Python port of run_spike_sz_pipeline_clean.m.

The MATLAB script's data wrangling, cohort construction, figures, tables, and
HTML output are reproduced in Python.  The MATLAB `fitglme(...,
Distribution="Binomial", Link="logit", (1|PatientID))` model is implemented
here as a binomial logistic random-intercept model fit by Gauss-Hermite
quadrature over the patient random intercept.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy import optimize, special, stats

import spike_phq2 as base


RNG = np.random.default_rng(1)

CANONICAL3 = ["General", "Temporal", "Frontal"]
MODEL_CATS = ["Temporal", "Frontal", "General"]
EPS_RATE = 30e-3
Y_LIMS = (-2, 4)
SPEARMAN_X_LIMS = (-3.5, 4)
SPEARMAN_Y_LIMS = (-1.5, 3)
ALLOWABLE_VISITS = base.ALLOWABLE_VISITS


def build_visit_level_table_r1(report: pd.DataFrame) -> pd.DataFrame:
    """Same as MATLAB build_visit_level_table_R1, including rule 2."""
    vuniq = base.build_visit_level_table_r1(report)
    mask_sz1 = np.isfinite(vuniq["Freq_R1"]) & (vuniq["Freq_R1"] > 0) & ~np.isfinite(vuniq["HasSz"])
    mask_sz0 = np.isfinite(vuniq["Freq_R1"]) & (vuniq["Freq_R1"] == 0) & ~np.isfinite(vuniq["HasSz"])
    vuniq.loc[mask_sz1, "HasSz"] = 1.0
    vuniq.loc[mask_sz0, "HasSz"] = 0.0
    print(
        f"[Rule 2] Imputed HasSz for {int(mask_sz1.sum())} visits with SzFreq>0, "
        f"{int(mask_sz0.sum())} visits with SzFreq=0"
    )
    return vuniq


def filter_outpatient_routine(spike: pd.DataFrame, report: pd.DataFrame, dur_col: str, max_routine_hours: float):
    n_patients_total = pd.to_numeric(report["patient_id"], errors="coerce").nunique()
    s, r = base.filter_outpatient_routine(spike, report, dur_col, max_routine_hours)
    return s, r, int(n_patients_total)


def build_filtered_view(sessions, report, typing_all, sz_freq, n_patients_total):
    views = base.build_filtered_view(sessions, report, typing_all, sz_freq)

    patients_kept = pd.to_numeric(sessions["Patient"], errors="coerce").dropna().drop_duplicates()
    n_after_outpt_routine = len(patients_kept)
    patient_level_pre = (
        sessions.assign(Patient=pd.to_numeric(sessions["Patient"], errors="coerce"))
        .groupby("Patient")
        .agg(MeanSpikeRate_perHour=("SpikeRate_perHour", base.matlab_mean))
        .reset_index()
        .merge(typing_all, on="Patient", how="inner")
    )
    etype_norm = patient_level_pre["EpilepsyType"].map(base._clean_scalar).str.lower()
    is_nesd = etype_norm.eq(base.NESD_LABEL.lower())
    is_bad = etype_norm.isin(base.BAD_TYPES) | etype_norm.eq("")
    is_epilepsy = ~(is_nesd | is_bad)
    patients_with_epilepsy = patient_level_pre.loc[is_epilepsy, ["Patient"]]
    sz_filtered = sz_freq.merge(pd.DataFrame({"Patient": patients_kept.astype(float)}), on="Patient", how="inner")
    sz_epi = sz_filtered.merge(patients_with_epilepsy, on="Patient", how="inner")
    n_with_epilepsy = sz_epi["Patient"].nunique()
    n_with_szfreq = sz_epi.loc[np.isfinite(sz_epi["MeanSzFreq"]), "Patient"].nunique()

    ec = SimpleNamespace(
        nTotal=n_patients_total,
        nAfterOutptRoutine=n_after_outpt_routine,
        nExcludedNoEpilepsy=n_after_outpt_routine - n_with_epilepsy,
        nExcludedNoSzFreq=n_with_epilepsy - n_with_szfreq,
        nFinalCohort=len(views.PatientLevelSpikeRates),
    )
    if ec.nExcludedNoEpilepsy + ec.nExcludedNoSzFreq + ec.nFinalCohort != ec.nAfterOutptRoutine:
        raise AssertionError("Flow count mismatch")
    views.ExclusionCounts = ec
    views.PatientTyping_AllEpilepsy = views.PatientTypingFiltered.copy()
    views.PatientTypingFiltered = views.PatientTypingFiltered[
        views.PatientTypingFiltered["EpiType3"].astype(str).isin(CANONICAL3)
    ].copy()
    print(f"[Cohort] {ec.nFinalCohort} epilepsy patients with documented seizure frequency")
    return views


def build_eeg_visit_pairs(vuniq, session_level, report_kept, patient_typing):
    session_level = session_level.copy()
    session_level["Patient"] = pd.to_numeric(session_level["Patient"], errors="coerce")
    session_level["Session"] = pd.to_numeric(session_level["Session"], errors="coerce")
    session_level["SpikesPerHour"] = pd.to_numeric(session_level["SpikesPerHour"], errors="coerce")
    eeg_dt = pd.to_datetime(
        report_kept["start_time_deid"].map(base._clean_scalar),
        errors="coerce",
    )
    eeg_dates = pd.DataFrame(
        {
            "Patient": pd.to_numeric(report_kept["Patient"], errors="coerce"),
            "Session": pd.to_numeric(report_kept["Session"], errors="coerce"),
            "EEG_Date": eeg_dt,
        }
    ).dropna()
    eeg_tbl = eeg_dates.merge(session_level[["Patient", "Session", "SpikesPerHour"]], on=["Patient", "Session"])
    vtyped = vuniq[["Patient", "VisitDate", "Freq_R1", "HasSz"]].merge(
        patient_typing[["Patient", "EpiType3"]], on="Patient", how="inner"
    )

    patients = np.intersect1d(eeg_tbl["Patient"].unique(), vtyped["Patient"].unique())
    chunks = []
    for patient in patients:
        eeg_rows = eeg_tbl[eeg_tbl["Patient"] == patient][["Patient", "Session", "EEG_Date", "SpikesPerHour"]]
        visit_rows = vtyped[vtyped["Patient"] == patient][["Patient", "VisitDate", "Freq_R1", "HasSz", "EpiType3"]]
        if eeg_rows.empty or visit_rows.empty:
            continue
        merged = eeg_rows.merge(visit_rows, on="Patient", how="cross" if False else "inner")
        # The inner merge above yields the patient-wise Cartesian product because
        # both sides contain only this one patient.
        chunks.append(merged)
    if chunks:
        pair = pd.concat(chunks, ignore_index=True)
        pair["SignedLag_days"] = (pair["VisitDate"] - pair["EEG_Date"]).dt.total_seconds() / 86400.0
        pair = pair.rename(columns={"Freq_R1": "SzFreq"})
        pair = pair[["Patient", "Session", "VisitDate", "SpikesPerHour", "SzFreq", "HasSz", "SignedLag_days", "EpiType3"]]
    else:
        pair = pd.DataFrame(columns=["Patient", "Session", "VisitDate", "SpikesPerHour", "SzFreq", "HasSz", "SignedLag_days", "EpiType3"])
    for col in ["Patient", "Session", "SpikesPerHour", "SzFreq", "HasSz", "SignedLag_days"]:
        pair[col] = pd.to_numeric(pair[col], errors="coerce")
    n_before = len(pair)
    keep = (
        np.isfinite(pair["SpikesPerHour"])
        & np.isfinite(pair["SzFreq"])
        & np.isfinite(pair["SignedLag_days"])
        & pair["EpiType3"].astype(str).str.len().gt(0)
        & pair["EpiType3"].isin(CANONICAL3)
    )
    pair = pair.loc[keep].copy()
    pair["EEG_ID"] = pair["Patient"].astype(str) + "_" + pair["Session"].astype(str)
    pair["LogSpikesPerHour"] = np.log(pair["SpikesPerHour"] + 1e-3)
    pair["SignedLag_years"] = pair["SignedLag_days"] / 365.25
    pair["PatientID"] = pair["Patient"].astype(str)
    print(f"[build_eeg_visit_pairs] {len(patients)} patients, {len(pair)} pairs ({n_before-len(pair)} removed)")
    return pair


def _design_matrix(t: pd.DataFrame, interactions: bool, signed: bool = False):
    x = pd.DataFrame(index=t.index)
    x["(Intercept)"] = 1.0
    x["LogSpikesPerHour"] = t["LogSpikesPerHour"].astype(float)
    if signed:
        x["SignedLag_years"] = t["SignedLag_years"].astype(float)
    else:
        x["AbsLag_years"] = t["AbsLag_years"].astype(float)
        x["LagDirection"] = t["LagDirection"].astype(float)
    epi = t["EpiType3_cat"].astype(str)
    x["EpiType3_cat_Frontal"] = (epi == "Frontal").astype(float)
    x["EpiType3_cat_General"] = (epi == "General").astype(float)
    if interactions:
        if signed:
            x["LogSpikesPerHour:SignedLag_years"] = x["LogSpikesPerHour"] * x["SignedLag_years"]
        else:
            x["LogSpikesPerHour:AbsLag_years"] = x["LogSpikesPerHour"] * x["AbsLag_years"]
            x["LogSpikesPerHour:LagDirection"] = x["LogSpikesPerHour"] * x["LagDirection"]
    return x


def _fit_logistic_re(y, x, groups, n_quad=15, start=None):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups)
    _, group_index = np.unique(groups, return_inverse=True)
    group_rows = [np.where(group_index == g)[0] for g in range(group_index.max() + 1)]
    z, w = hermgauss(n_quad)
    log_w = np.log(w) - 0.5 * np.log(np.pi)
    p = x.shape[1]
    if start is None:
        start = np.zeros(p + 1)
        start[-1] = -0.3

    def nll(theta):
        beta = theta[:p]
        sigma = np.exp(theta[p])
        total = 0.0
        for rows in group_rows:
            xb = x[rows] @ beta
            vals = []
            for zi, lwi in zip(z, log_w):
                eta = xb + math.sqrt(2) * sigma * zi
                ll = np.sum(y[rows] * eta - np.logaddexp(0, eta))
                vals.append(lwi + ll)
            total += special.logsumexp(vals)
        return -total

    res = optimize.minimize(nll, start, method="BFGS", options={"maxiter": 1000, "gtol": 1e-5})
    hess_inv = np.asarray(res.hess_inv) if hasattr(res, "hess_inv") else np.full((p + 1, p + 1), np.nan)
    return SimpleNamespace(
        params=res.x[:p],
        log_sigma=res.x[p],
        sigma=float(np.exp(res.x[p])),
        success=bool(res.success),
        message=res.message,
        nll=float(res.fun),
        cov=hess_inv[:p, :p] if hess_inv.shape[0] >= p else np.full((p, p), np.nan),
    )


def _fe_table(terms, fit, alpha):
    beta = fit.params
    se = np.sqrt(np.maximum(np.diag(fit.cov), 0)) if fit.cov.size else np.full_like(beta, np.nan)
    z = beta / se
    p = 2 * stats.norm.sf(np.abs(z))
    return pd.DataFrame(
        {
            "Term": terms,
            "Beta": beta,
            "SE": se,
            "t": z,
            "p": p,
            "OR": np.exp(beta),
            "OR_lo": np.exp(beta - stats.norm.ppf(1 - alpha / 2) * se),
            "OR_hi": np.exp(beta + stats.norm.ppf(1 - alpha / 2) * se),
        }
    )


def _bootstrap_model(t, terms, interactions, signed, n_boot, alpha, label, fit):
    if n_boot == 0 or fit is None:
        return pd.DataFrame(), np.empty((0, len(terms))), 0, 0
    patients = t["PatientID"].unique()
    boot = np.full((n_boot, len(terms)), np.nan)
    print(f"\nBootstrapping {label} ({n_boot} iterations)...")
    for b in range(n_boot):
        sampled = RNG.choice(patients, size=len(patients), replace=True)
        chunks = []
        for k, pat in enumerate(sampled):
            chunk = t[t["PatientID"] == pat].copy()
            chunk["PatientID"] = str(k)
            chunks.append(chunk)
        tb = pd.concat(chunks, ignore_index=True)
        xb = _design_matrix(tb, interactions=interactions, signed=signed)
        try:
            fb = _fit_logistic_re(tb["HasSz_bin"], xb, tb["PatientID"], start=np.r_[fit.params, fit.log_sigma])
            if fb.success:
                boot[b, :] = fb.params
        except Exception:
            pass
    converged = np.all(np.isfinite(boot), axis=1)
    boot = boot[converged]
    print(f"{label} bootstrap: {boot.shape[0]}/{n_boot} converged ({100*boot.shape[0]/max(1,n_boot):.1f}%)")
    if boot.size == 0:
        return pd.DataFrame(), boot, 0, n_boot
    ci_lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    ci_hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    boot_p = [min(2 * min(np.mean(boot[:, k] <= 0), np.mean(boot[:, k] >= 0)), 1) for k in range(boot.shape[1])]
    table = pd.DataFrame(
        {
            "Term": terms,
            "Beta": fit.params,
            "Boot_CI_lo": ci_lo,
            "Boot_CI_hi": ci_hi,
            "OR": np.exp(fit.params),
            "OR_CI_lo": np.exp(ci_lo),
            "OR_CI_hi": np.exp(ci_hi),
            "Boot_p": boot_p,
        }
    )
    return table, boot, int(boot.shape[0]), int(n_boot)


def fit_mixed_effects_models(pair: pd.DataFrame, n_boot: int, alpha: float, quad_nodes: int):
    keep = (
        pair["EpiType3"].isin(["Frontal", "General", "Temporal"])
        & np.isfinite(pair["LogSpikesPerHour"])
        & np.isfinite(pair["SignedLag_years"])
        & np.isfinite(pair["HasSz"])
    )
    t = pair.loc[keep].copy()
    t["HasSz_bin"] = (t["HasSz"] == 1).astype(float)
    t["LogSzFreq"] = np.log(t["SzFreq"] + 1e-3)
    t["AbsLag_years"] = np.abs(t["SignedLag_years"])
    t["LagDirection"] = np.sign(t["SignedLag_years"])
    t.loc[t["SignedLag_years"] == 0, "LagDirection"] = 1
    t["EpiType3_cat"] = pd.Categorical(t["EpiType3"], categories=MODEL_CATS)
    print(f"[Model table] {len(t)} pairs, {t['Patient'].nunique()} patients")

    print("\nFitting M1 (logistic + subtypes + interactions)...")
    x1 = _design_matrix(t, interactions=True, signed=False)
    fit1 = _fit_logistic_re(t["HasSz_bin"], x1, t["PatientID"], n_quad=quad_nodes)
    print(f"M1 {'converged' if fit1.success else 'finished with warning'}: {fit1.message}")

    print("\nFitting M2 (logistic + subtypes, no interactions)...")
    x2 = _design_matrix(t, interactions=False, signed=False)
    fit2 = _fit_logistic_re(t["HasSz_bin"], x2, t["PatientID"], n_quad=quad_nodes)
    print(f"M2 {'converged' if fit2.success else 'finished with warning'}: {fit2.message}")

    fe1 = _fe_table(list(x1.columns), fit1, alpha)
    fe2 = _fe_table(list(x2.columns), fit2, alpha)
    lr = 2 * (-(fit1.nll) - (-(fit2.nll)))
    lrt_p = float(stats.chi2.sf(lr, max(1, len(x1.columns) - len(x2.columns))))
    print(f"\nLRT M1 vs M2: chi2={lr:.3f}, p={lrt_p:.4g}")

    t_after = t[t["SignedLag_years"] >= 0].copy()
    t_before = t[t["SignedLag_years"] <= 0].copy()
    dir_fits = {}
    dir_tables = {}
    for label, tt in [("after", t_after), ("before", t_before)]:
        print(f"\n[Directional] {label}: {len(tt)} pairs, {tt['Patient'].nunique()} patients")
        xd = _design_matrix(tt, interactions=True, signed=True)
        fd = _fit_logistic_re(tt["HasSz_bin"], xd, tt["PatientID"], n_quad=quad_nodes) if len(tt) else None
        dir_fits[label] = fd
        dir_tables[label] = _fe_table(list(xd.columns), fd, alpha) if fd else pd.DataFrame()

    bt1, bb1, nc1, nt1 = _bootstrap_model(t, list(x1.columns), True, False, n_boot, alpha, "M1", fit1)
    bt2, bb2, nc2, nt2 = _bootstrap_model(t, list(x2.columns), False, False, n_boot, alpha, "M2", fit2)
    bta, bba, nca, nta = _bootstrap_model(t_after, list(_design_matrix(t_after, True, True).columns), True, True, n_boot, alpha, "M_after", dir_fits["after"])
    btb, bbb, ncb, ntb = _bootstrap_model(t_before, list(_design_matrix(t_before, True, True).columns), True, True, n_boot, alpha, "M_before", dir_fits["before"])

    return SimpleNamespace(
        ModelTable=t,
        mdl_M1=fit1,
        mdl_M2=fit2,
        FE_M1=fe1,
        FE_M2=fe2,
        BootstrapTable1=bt1,
        BootstrapBetas1=bb1,
        BootstrapTable2=bt2,
        BootstrapBetas2=bb2,
        LRT_p=lrt_p,
        mdl_after=dir_fits["after"],
        mdl_before=dir_fits["before"],
        FE_after=dir_tables["after"],
        FE_before=dir_tables["before"],
        BootstrapTable_after=bta,
        BootstrapBetas_after=bba,
        BootstrapTable_before=btb,
        BootstrapBetas_before=bbb,
        BootstrapConvergence=SimpleNamespace(
            M1_nConverged=nc1,
            M1_nTotal=nt1,
            M2_nConverged=nc2,
            M2_nTotal=nt2,
            after_nConverged=nca,
            after_nTotal=nta,
            before_nConverged=ncb,
            before_nTotal=ntb,
        ),
    )


def _coef(mmr, term):
    row = mmr.FE_M1[mmr.FE_M1["Term"] == term]
    return float(row["Beta"].iloc[0]) if not row.empty else 0.0


def make_model_figure(mmr, out_path: Path):
    bt = mmr.BootstrapTable1 if len(mmr.BootstrapTable1) else pd.DataFrame()
    fe = mmr.FE_M1.copy()
    rows = []
    for _, r in fe.iterrows():
        if r.Term == "(Intercept)":
            continue
        br = bt[bt["Term"] == r.Term] if len(bt) else pd.DataFrame()
        if len(br):
            lo, hi, pval = br["OR_CI_lo"].iloc[0], br["OR_CI_hi"].iloc[0], br["Boot_p"].iloc[0]
        else:
            lo, hi, pval = r.OR_lo, r.OR_hi, r.p
        rows.append((r.Term, r.OR, lo, hi, pval))
    plot = pd.DataFrame(rows, columns=["Term", "OR", "lo", "hi", "p"])
    labels = {
        "LogSpikesPerHour": "Log spike rate",
        "AbsLag_years": "EEG-visit gap (years)",
        "LagDirection": "Visit after vs before EEG",
        "EpiType3_cat_Frontal": "Frontal vs Temporal",
        "EpiType3_cat_General": "Generalized vs Temporal",
        "LogSpikesPerHour:AbsLag_years": "Spike rate effect per year of gap",
        "LogSpikesPerHour:LagDirection": "Spike rate effect: visit before or after",
    }
    plot["Label"] = plot["Term"].map(labels).fillna(plot["Term"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    y = np.arange(len(plot))[::-1]
    for yi, (_, r) in zip(y, plot.iterrows()):
        col = "#1a4db3" if r.p < 0.05 else "0.6"
        ax1.plot([r.lo, r.hi], [yi, yi], color=col, lw=2.5)
        ax1.scatter([r.OR], [yi], color=col, s=80)
        ax1.text(r.hi + 0.01, yi, base.p_label(r.p), va="center", fontsize=10)
    ax1.axvline(1, color="k", ls="--")
    ax1.set_yticks(y, plot["Label"])
    ax1.set_xlabel("Odds Ratio")
    ax1.set_title("A. Spike rate, epilepsy type, and EEG-visit gap\npredict seizure occurrence", fontweight="bold")
    ax1.grid(True, alpha=0.25)

    spike_raw = np.linspace(0, 50, 200)
    log_spike = np.log(spike_raw + 1e-3)
    b0 = _coef(mmr, "(Intercept)")
    b_sp = _coef(mmr, "LogSpikesPerHour")
    b_lag = _coef(mmr, "AbsLag_years")
    b_dir = _coef(mmr, "LagDirection")
    b_int_lag = _coef(mmr, "LogSpikesPerHour:AbsLag_years")
    b_int_dir = _coef(mmr, "LogSpikesPerHour:LagDirection")
    for lag, lab, col in [(0.5, "6 months", "#0d4db3"), (2, "2 years", "#267fcc"), (4, "4 years", "#66a6d9")]:
        eta = b0 + b_sp * log_spike + b_lag * lag + b_dir + b_int_lag * log_spike * lag + b_int_dir * log_spike
        ax2.plot(spike_raw, special.expit(eta), label=lab, color=col, lw=2.5)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0.3, 0.61)
    ax2.set_xlabel("Spike rate (spikes/hour)")
    ax2.set_ylabel("P(seizure reported at visit)")
    ax2.set_title("B. Spike rates are most predictive\nwhen EEG is obtained close to the visit", fontweight="bold")
    ax2.legend(title="EEG-visit lag", loc="lower right")
    ax2.grid(True, alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved main model figure: {out_path}")


def make_figsup_lag(mmr, vuniq, report, out_path: Path):
    t = mmr.ModelTable
    ref = pd.Timestamp(2000, 1, 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    cohort = t["Patient"].unique()
    vc = vuniq[vuniq["Patient"].isin(cohort) & vuniq["HasSz"].isin([0, 1])].copy()
    vc["YearsSinceFirst"] = (vc["VisitDate"] - ref).dt.days / 365.25
    edges = np.array([0, 1, 2, 3, 4])
    centers = (edges[:-1] + edges[1:]) / 2
    prop = []
    med = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = vc[(vc["YearsSinceFirst"] >= lo) & (vc["YearsSinceFirst"] < hi)]
        prop.append(np.nanmean(rows["HasSz"]) if len(rows) >= 10 else np.nan)
        med.append(np.nanmedian(rows["Freq_R1"]) if len(rows) >= 10 else np.nan)
    ax1.plot(centers, prop, "o-", color="#cc4d1a", label="Proportion with seizures")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Years after first visit")
    ax1.set_ylabel("Proportion with seizures")
    ax1b = ax1.twinx()
    ax1b.plot(centers, np.log10(np.asarray(med) + 1e-3), "s--", color="#1a8c8c", label="Median sz/month")
    ax1b.set_ylabel("Median sz/month (log scale)")
    base.set_log10_ticks(ax1b, "y", 1e-3, (-2, 2))
    ax1.set_title("A. Seizure burden tends to decrease over time", fontweight="bold")
    ax2.hist(t["AbsLag_years"], bins=40, weights=np.ones(len(t)) / max(1, len(t)), color="0.3", alpha=0.6)
    ax2.axvline(1, color="k", ls="--")
    ax2.set_xlabel("Absolute EEG-visit gap (years)")
    ax2.set_ylabel("Proportion of pairs")
    ax2.set_title("B. EEG and visit are often separated by years", fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved supplemental lag figure: {out_path}")


def make_flowchart_figure(views, mmr, out_path: Path):
    ec = views.ExclusionCounts
    n_subtype = mmr.ModelTable["Patient"].nunique()
    n_pairs = len(mmr.ModelTable)
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.axis("off")
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1)
    boxes = [
        (0.46, 0.90, f"All patients with EEG data\nN = {ec.nTotal}", "#3973b3"),
        (0.46, 0.74, f"Outpatient routine EEG <=4 hours\nN = {ec.nAfterOutptRoutine}", "#3973b3"),
        (0.46, 0.57, f"LLM-confirmed epilepsy diagnosis\nN = {ec.nAfterOutptRoutine - ec.nExcludedNoEpilepsy}", "#3973b3"),
        (0.46, 0.40, f"Documented seizure frequency\nN = {ec.nFinalCohort} (primary cohort)", "#268c66"),
        (0.46, 0.18, f"Known epilepsy subtype\n(temporal, frontal, generalized)\nfor mixed effects model\nN = {n_subtype} patients, {n_pairs} EEG-visit pairs", "#268c66"),
    ]
    for x, y, txt, col in boxes:
        ax.add_patch(plt.Rectangle((x - 0.26, y - 0.04), 0.52, 0.08, facecolor=col, edgecolor=col, alpha=0.25, lw=1.8))
        ax.text(x, y, txt, ha="center", va="center", fontsize=11)
    for y1, y2 in [(0.86, 0.78), (0.70, 0.61), (0.53, 0.44), (0.36, 0.24)]:
        ax.annotate("", xy=(0.46, y2), xytext=(0.46, y1), arrowprops=dict(arrowstyle="-|>", color="0.3"))
    exclusions = [
        ((0.90, 0.82), f"Excluded: inpatient or\nambulatory EEG\nN = {ec.nTotal - ec.nAfterOutptRoutine}"),
        ((0.90, 0.655), f"Excluded: no epilepsy\ndiagnosis (NESD, uncertain,\nor unknown)\nN = {ec.nExcludedNoEpilepsy}"),
        ((0.90, 0.485), f"Excluded: no documented\nseizure frequency\nN = {ec.nExcludedNoSzFreq}"),
    ]
    for (x, y), txt in exclusions:
        ax.annotate("", xy=(x - 0.18, y), xytext=(0.46, y), arrowprops=dict(arrowstyle="-|>", color="#cc4d1a"))
        ax.add_patch(plt.Rectangle((x - 0.17, y - 0.04), 0.34, 0.08, facecolor="#cc4d1a", edgecolor="#cc4d1a", alpha=0.25, lw=1.5))
        ax.text(x, y, txt, ha="center", va="center", fontsize=9)
    ax.text(0.46, 0.97, "Study participant flow", ha="center", va="top", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved flow diagram: {out_path}")


def clean_term(raw: str) -> str:
    repl = {
        "(Intercept)": "Intercept",
        "LogSpikesPerHour:AbsLag_years": "Log spike rate x Absolute lag",
        "LogSpikesPerHour:LagDirection": "Log spike rate x Lag direction",
        "LogSpikesPerHour": "Log spike rate",
        "AbsLag_years": "Absolute lag (years)",
        "LagDirection": "Lag direction (after vs before)",
        "SignedLag_years": "Signed lag (years)",
        "EpiType3_cat_Frontal": "Frontal vs Temporal epilepsy",
        "EpiType3_cat_General": "Generalized vs Temporal epilepsy",
    }
    return repl.get(raw, raw)


def write_table_s1(mmr, out_path: Path):
    rows = []
    for model_label, fe, bt in [
        ("M1 (logistic, subtypes, interactions)", mmr.FE_M1, mmr.BootstrapTable1),
        ("M2 (logistic, subtypes, no interactions)", mmr.FE_M2, mmr.BootstrapTable2),
    ]:
        for _, r in fe.iterrows():
            if r.Term == "(Intercept)":
                continue
            p_val = r.p
            lo, hi, ci_src = r.OR_lo, r.OR_hi, "Quadrature Wald"
            br = bt[bt["Term"] == r.Term] if len(bt) else pd.DataFrame()
            if len(br):
                p_val = br["Boot_p"].iloc[0]
                lo, hi, ci_src = br["OR_CI_lo"].iloc[0], br["OR_CI_hi"].iloc[0], "Bootstrap"
            rows.append((model_label, clean_term(r.Term), f"{r.OR:.3f}", f"{lo:.3f}", f"{hi:.3f}", ci_src, "<0.001" if p_val < 0.001 else f"{p_val:.3f}" if p_val < 0.01 else f"{p_val:.2f}"))
    out = pd.DataFrame(rows, columns=["Model", "Term", "Estimate", "CI_lower", "CI_upper", "CI_method", "p_value"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def build_table1_flat(views, sz_freq, vuniq, n_boot, alpha):
    table1 = base.build_table1_flat(views, sz_freq, vuniq, n_boot, alpha).copy()
    table1.loc[table1["Variable"] == "Total N patients with >=1 outpatient routine EEG", "Variable"] = "Total N patients"

    all_patients = views.PatientLevelSpikeRates["Patient"].to_numpy(float)
    v_cohort = vuniq[vuniq["Patient"].isin(all_patients)].copy()
    follow = v_cohort.groupby("Patient")["VisitDate"].agg(lambda d: (d.max() - d.min()).days / 365.25).reset_index(name="FollowupYears")
    follow = pd.DataFrame({"Patient": all_patients}).merge(follow, on="Patient", how="left")["FollowupYears"].to_numpy(float)
    doc = v_cohort.groupby("Patient")["Freq_R1"].agg(lambda f: np.mean(np.isfinite(f))).reset_index(name="FracDocumented")
    doc = pd.DataFrame({"Patient": all_patients}).merge(doc, on="Patient", how="left")["FracDocumented"].to_numpy(float)

    rs = base.resolve_reported_spike_status(views.ReportForKeptSessions)
    pat = rs.groupby("Patient")["ReportStatus"].agg(
        HasPresent=lambda x: np.any(x.astype(str) == "present"),
        HasAbsent=lambda x: np.any(x.astype(str) == "absent"),
    ).reset_index()
    status = np.array(["unknown"] * len(pat), dtype=object)
    status[pat["HasAbsent"].to_numpy() & ~pat["HasPresent"].to_numpy()] = "absent"
    status[pat["HasPresent"].to_numpy()] = "present"
    n_pre = int(np.sum(status == "present"))
    n_abs = int(np.sum(status == "absent"))
    n_unk = int(np.sum(status == "unknown"))
    n_pat = len(status)

    def row(var, stat):
        return pd.DataFrame({"Variable": [var], "Statistic": [stat]})

    clinic_idx = table1.index[table1["Variable"] == "Number of clinic visits"]
    if len(clinic_idx):
        i = int(clinic_idx[0]) + 1
        extra = pd.DataFrame(
            {
                "Variable": ["Follow-up duration (years)", "Visits with documented seizure frequency"],
                "Statistic": [
                    f"{np.nanmedian(follow):.1f} ({np.nanpercentile(follow,25):.1f}-{np.nanpercentile(follow,75):.1f})",
                    f"{np.nanmedian(doc)*100:.1f}% ({np.nanpercentile(doc,25)*100:.1f}-{np.nanpercentile(doc,75)*100:.1f})",
                ],
            }
        )
        table1 = pd.concat([table1.iloc[:i], extra, table1.iloc[i:]], ignore_index=True)

    table1.loc[table1["Variable"] == "Reported spikes", ["Variable", "Statistic"]] = ["EEGs with reported spikes", "N (% EEGs)"]
    patient_rows = pd.DataFrame(
        {
            "Variable": ["Patients with reported spikes", "    Present", "    Absent", "    Unknown"],
            "Statistic": [
                "N (% patients)",
                f"{n_pre} ({100*n_pre/max(1,n_pat):.1f}%)",
                f"{n_abs} ({100*n_abs/max(1,n_pat):.1f}%)",
                f"{n_unk} ({100*n_unk/max(1,n_pat):.1f}%)",
            ],
        }
    )
    return pd.concat([table1, patient_rows], ignore_index=True)


def write_results_html(out_path, views, sz_freq, fig1stats, sp_main, sp_s1, report, mmr, vuniq, nearfar):
    spearman_results, rs_all, p_all, n_all, rho_lo, rho_hi, subtype_ci = sp_main
    ec = views.ExclusionCounts
    pl = views.PatientLevelSpikeRates
    sf = pd.DataFrame({"Patient": pl["Patient"]}).merge(sz_freq, on="Patient")["MeanSzFreq"].to_numpy(float)
    sr = pl["MeanSpikeRate_perHour"].to_numpy(float)
    sf_med, sf_lo, sf_hi = base.bootstrap_median_ci(sf, 5000, 0.05)
    sr_med, sr_lo, sr_hi = base.bootstrap_median_ci(sr, 5000, 0.05)
    rs = base.resolve_reported_spike_status(report)
    pat_spike = rs.groupby("Patient")["ReportStatus"].apply(lambda x: np.any(x.astype(str) == "present"))
    bt = mmr.BootstrapTable1 if len(mmr.BootstrapTable1) else mmr.FE_M1.rename(columns={"OR_lo": "OR_CI_lo", "OR_hi": "OR_CI_hi"})
    def row(term):
        return bt[bt["Term"] == term].iloc[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write('<html><head><meta charset="UTF-8"><title>Results</title></head><body>\n')
        f.write("<h2>Cohort summary</h2>\n")
        f.write(f"<p>Of {ec.nTotal} patients with EEG data, {ec.nTotal - ec.nAfterOutptRoutine} were excluded because their EEG was not outpatient routine, {ec.nExcludedNoEpilepsy} were excluded without epilepsy, and {ec.nExcludedNoSzFreq} were excluded without documented seizure frequency, yielding {ec.nFinalCohort} patients with {len(report)} EEGs. {int(pat_spike.sum())} patients ({100*pat_spike.mean():.1f}%) had spikes reported on at least one EEG. Median [95% CI] monthly seizure frequency was {sf_med:.2f} [{sf_lo:.2f}-{sf_hi:.2f}], and median spikes/hour was {sr_med:.2f} [{sr_lo:.2f}-{sr_hi:.2f}].</p>\n")
        f.write("<h2>Spike rates by patient groups</h2>\n")
        f.write(f"<p>Spike rates were higher in EEGs with clinically-reported spikes (median {fig1stats.m_pre:.2f} [95% CI {fig1stats.lo_pre:.2f}-{fig1stats.hi_pre:.2f}]) than without ({fig1stats.m_abs:.2f} [{fig1stats.lo_abs:.2f}-{fig1stats.hi_abs:.2f}]) ({base.format_p_html(fig1stats.p_rankSum_A)}, Cliff's &delta;={fig1stats.effectA_cliff:.2f}).</p>\n")
        f.write("<h2>Spike rate and seizure frequency</h2>\n")
        f.write(f"<p>Spike rate and seizure frequency were positively correlated (N={n_all}, rho={rs_all:.2f} [95% CI {rho_lo:.2f}-{rho_hi:.2f}], {base.format_p_html(p_all)}).</p>\n")
        f.write("<h2>Mixed effects model</h2>\n")
        r_spike = row("LogSpikesPerHour")
        f.write(f"<p>The logistic random-intercept model used {len(mmr.ModelTable)} EEG-visit pairs from {mmr.ModelTable.Patient.nunique()} patients. Higher spike rates were associated with seizure reporting (OR={r_spike.OR:.2f} [95% CI {r_spike.OR_CI_lo:.2f}-{r_spike.OR_CI_hi:.2f}]). Interaction terms jointly improved model fit versus the no-interaction model ({base.format_p_html(mmr.LRT_p)}).</p>\n")
        f.write("<h2>Bootstrap diagnostics</h2>\n")
        bc = mmr.BootstrapConvergence
        f.write(f"<p>M1: {bc.M1_nConverged}/{bc.M1_nTotal} bootstrap iterations converged. M2: {bc.M2_nConverged}/{bc.M2_nTotal} converged.</p>\n")
        f.write("<h2>Figure S5 legend</h2>\n")
        f.write(f"<p>Tertile cutoffs were {nearfar.nearDays:.0f} and {nearfar.farDays:.0f} days. N={nearfar.nPatients}; rho_short={nearfar.rho_near:.2f}; rho_long={nearfar.rho_far:.2f}; delta rho={nearfar.delta_obs:.3f} [{nearfar.delta_ci_lo:.2f}-{nearfar.delta_ci_hi:.2f}], one-sided p={nearfar.p_one_sided:.3f}.</p>\n")
        f.write("</body></html>\n")


def compute_visit_eeg_gaps(vuniq, report):
    eeg_dt = pd.to_datetime(report["start_time_deid"].map(base._clean_scalar), errors="coerce")
    eeg = pd.DataFrame({"Patient": pd.to_numeric(report["Patient"], errors="coerce"), "EEG_Date": eeg_dt}).dropna()
    gap = vuniq[["Patient", "VisitDate"]].copy()
    vals = []
    for _, row in gap.iterrows():
        dates = eeg.loc[eeg["Patient"] == row["Patient"], "EEG_Date"]
        if dates.empty or pd.isna(row["VisitDate"]):
            vals.append(np.nan)
        else:
            vals.append(float(np.min(np.abs((dates - row["VisitDate"]).dt.total_seconds() / 86400))))
    gap["MinAbsGap_days"] = vals
    gap["MinAbsGap_years"] = gap["MinAbsGap_days"] / 365.25
    return gap


def restrict_visits_by_min_abs_gap(vuniq, report, min_days, max_days):
    gap = compute_visit_eeg_gaps(vuniq, report)
    vf = vuniq.merge(gap[["Patient", "VisitDate", "MinAbsGap_days"]], on=["Patient", "VisitDate"], how="left")
    keep = (vf["MinAbsGap_days"] >= min_days) & (vf["MinAbsGap_days"] <= max_days)
    print(f"[Visit-EEG distance] Kept {int(keep.sum())}/{len(vuniq)} visits with min|gap| in [{min_days:g}, {max_days:g}] days")
    return vf.loc[keep, vuniq.columns].copy()


def plot_delta_rho_histogram(views, vuniq, report, near_q, far_q, n_boot, alpha, out_png: Path):
    base_patients = np.asarray(sorted(views.PatientSpikeSz_All["Patient"].unique()), dtype=float)
    spike_tbl = views.PatientLevelSpikeRates[["Patient", "MeanSpikeRate_perHour"]].merge(pd.DataFrame({"Patient": base_patients}), on="Patient")
    v_base = vuniq.merge(pd.DataFrame({"Patient": base_patients}), on="Patient")
    v_base = v_base[np.isfinite(v_base["Freq_R1"])]
    visits_per = v_base.groupby("Patient")["Freq_R1"].size()
    print(f"[Visit counts] Patients with >=2 clinic visits with documented seizure frequency: {(visits_per >= 2).sum()}/{len(visits_per)} ({100 * (visits_per >= 2).sum() / max(1,len(visits_per)):.1f}%)")
    eeg_per = report.groupby("Patient")["Session"].nunique()
    print(f"Patients with >=2 EEGs: {(eeg_per >= 2).sum()}/{len(eeg_per)} ({100 * (eeg_per >= 2).sum() / max(1,len(eeg_per)):.1f}%)")
    gaps = compute_visit_eeg_gaps(vuniq, report)["MinAbsGap_days"].to_numpy(float)
    gaps = gaps[np.isfinite(gaps)]
    if len(gaps) == 0:
        raise ValueError("No finite MinAbsGap_days found.")
    near_days, far_days = float(np.quantile(gaps, near_q)), float(np.quantile(gaps, far_q))
    v_near = restrict_visits_by_min_abs_gap(vuniq, report, 0, near_days)
    v_far = restrict_visits_by_min_abs_gap(vuniq, report, far_days, np.inf)
    sz_near = base.build_patient_seizure_metrics(v_near)[["Patient", "MeanSzFreq"]].rename(columns={"MeanSzFreq": "MeanSzFreq_near"})
    sz_far = base.build_patient_seizure_metrics(v_far)[["Patient", "MeanSzFreq"]].rename(columns={"MeanSzFreq": "MeanSzFreq_far"})
    j = spike_tbl.merge(sz_near, on="Patient").merge(sz_far, on="Patient")
    j = j[np.isfinite(j["MeanSpikeRate_perHour"]) & np.isfinite(j["MeanSzFreq_near"]) & np.isfinite(j["MeanSzFreq_far"])]
    if len(j) < 3:
        raise ValueError(f"Not enough patients with BOTH near and far seizure metrics (n={len(j)}).")
    x = j["MeanSpikeRate_perHour"].to_numpy(float)
    yn = j["MeanSzFreq_near"].to_numpy(float)
    yf = j["MeanSzFreq_far"].to_numpy(float)
    rho_near, _ = base.spearman(x, yn)
    rho_far, _ = base.spearman(x, yf)
    delta_obs = rho_near - rho_far
    delta = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, len(j), len(j))
        rn, _ = base.spearman(x[idx], yn[idx])
        rf, _ = base.spearman(x[idx], yf[idx])
        delta[b] = rn - rf
    ci_lo, ci_hi = np.nanpercentile(delta, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_one = float(np.nanmean(delta <= 0))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.8), constrained_layout=True)
    ax1.hist(gaps, bins=60)
    ax1.axvline(near_days, color="k", ls="--", lw=2)
    ax1.axvline(far_days, color="k", ls="--", lw=2)
    ax1.set_xlabel("|Visit - EEG| gap (days)")
    ax1.set_ylabel("Visit count")
    ax1.set_title("A. Visit-EEG gap distribution with lower and upper third cutoffs")
    ax2.hist(delta, bins=40)
    ax2.axvline(0, color="k", ls="--", lw=2)
    ax2.axvline(np.nanmedian(delta), color="k", lw=2)
    max_abs = np.nanmax(np.abs(delta))
    max_abs = max_abs if np.isfinite(max_abs) and max_abs else 1e-3
    ax2.set_xlim(-1.08 * max_abs, 1.08 * max_abs)
    ax2.set_xlabel("Delta rho = rho_short gap - rho_long gap")
    ax2.set_ylabel("Bootstrap count")
    ax2.set_title(f"B. Distribution of differences in spike-seizure correlation\nbetween short and long visit-EEG gaps\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}], p = {p_one:.3g}")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved tertile figure: {out_png}")
    return SimpleNamespace(
        nPatients=len(j), nearQ=near_q, farQ=far_q, nearDays=near_days, farDays=far_days,
        rho_near=rho_near, rho_far=rho_far, delta_obs=delta_obs, delta_boot=delta,
        delta_median=float(np.nanmedian(delta)), delta_ci_lo=float(ci_lo), delta_ci_hi=float(ci_hi),
        p_one_sided=p_one, p_two_sided=2 * min(p_one, float(np.nanmean(delta >= 0))),
        tableUsed=j, gapsUsed=gaps,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Python port of run_spike_sz_pipeline_clean.m")
    p.add_argument("--spike-csv", type=Path, default=Path("../data/spike_counts.csv"))
    p.add_argument("--report-csv", type=Path, default=Path("../data/clinical_data_deidentified.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("../output"))
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--max-routine-hours", type=float, default=4)
    p.add_argument("--quad-nodes", type=int, default=9, help="Gauss-Hermite quadrature nodes for the logistic random-intercept model")
    return p.parse_args()


def run_pipeline(args):
    spike = base.read_csv_stringy(args.spike_csv)
    base.require_cols(spike, ["Patient", "Session", "count_0_46", "Duration_sec"], "SpikeSummaryTable")
    spike["count_0_46"] = base.numeric(spike["count_0_46"])
    spike["Duration_sec"] = base.numeric(spike["Duration_sec"])
    spike["SpikeRate_perHour"] = spike["count_0_46"] / spike["Duration_sec"] * 3600

    report = base.read_csv_stringy(args.report_csv)
    base.require_cols(
        report,
        [
            "patient_id", "session_number", "acquired_on", "report_PATIENT_CLASS", "jay_in_or_out",
            "visit_type", "visit_dates_deid", "sz_freqs", "visit_hasSz", "epilepsy_type",
            "epilepsy_specific", "nlp_gender", "deid_birth_date", "start_time_deid",
            "report_SPORADIC_EPILEPTIFORM_DISCHARGES", "jay_focal_epi", "jay_multifocal_epi", "jay_gen_epi",
        ],
        "ReportTable",
    )
    report = base.filter_visit_arrays_by_type(report, ALLOWABLE_VISITS)
    spike, report, n_total = filter_outpatient_routine(spike, report, "Duration_sec", args.max_routine_hours)
    spike["Patient"] = pd.to_numeric(spike["Patient"], errors="coerce")
    spike["Session"] = pd.to_numeric(spike["Session"], errors="coerce")
    report["patient_id"] = pd.to_numeric(report["patient_id"], errors="coerce")
    report["session_number"] = pd.to_numeric(report["session_number"], errors="coerce")
    base.assert_unique_keys(spike, "Patient", "Session", "SpikeSummaryTable")
    base.assert_unique_keys(report, "patient_id", "session_number", "ReportTable")

    vuniq = build_visit_level_table_r1(report)
    typing = base.build_patient_typing_from_report(report, CANONICAL3)
    sz_freq = base.build_patient_seizure_metrics(vuniq)
    views = build_filtered_view(spike, report, typing, sz_freq, n_total)

    pair = build_eeg_visit_pairs(vuniq, views.SessionLevelSpikeRates, views.ReportForKeptSessions, views.PatientTypingFiltered)
    print(f"Canonical-subtype pairs: {len(pair)}, patients: {pair.Patient.nunique()}")
    mmr = fit_mixed_effects_models(pair, args.n_boot, 0.05, args.quad_nodes)

    out = args.output_dir
    plot_boot = max(1, args.n_boot)
    make_flowchart_figure(views, mmr, out / "FigFlow.png")
    fig1stats = base.make_fig1_controls(views, out / "Fig1.png", plot_boot, 0.05)
    print(f"Saved Fig 1: {out / 'Fig1.png'}")
    sp_main = base.spearman_plotting_function(views.PatientSpikeSz_All, views.PatientSpikeSz_Typed, out / "Fig2.png", "MeanSzFreq", "", False, plot_boot, 0.05)
    sp_s1 = base.spearman_plotting_function(views.PatientSpikeSz_All, views.PatientSpikeSz_Typed, out / "FigS1.png", "MeanSzFreq", " (positive spike/seizures only)", True, plot_boot, 0.05)
    make_model_figure(mmr, out / "FigModel.png")
    make_figsup_lag(mmr, vuniq, views.ReportForKeptSessions, out / "FigSupLag.png")
    base.make_figs2_sz_by_reported_spikes(views, sz_freq, out / "FigS2.png", plot_boot, 0.05)
    nearfar = plot_delta_rho_histogram(views, vuniq, views.ReportForKeptSessions, 0.333, 0.667, plot_boot, 0.05, out / "FigSTertile.png")

    table1 = build_table1_flat(views, sz_freq, vuniq, plot_boot, 0.05)
    table1.to_csv(out / "Table1.csv", index=False)
    print(f"Wrote Table 1: {out / 'Table1.csv'}")
    write_table_s1(mmr, out / "TableS1.csv")
    print(f"Wrote Table S1: {out / 'TableS1.csv'}")
    write_results_html(out / "results_summary.html", views, sz_freq, fig1stats, sp_main, sp_s1, views.ReportForKeptSessions, mmr, vuniq, nearfar)
    print(f"Wrote HTML: {out / 'results_summary.html'}")


if __name__ == "__main__":
    run_pipeline(parse_args())
