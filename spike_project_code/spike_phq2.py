#!/usr/bin/env python3
"""
Python port of spike_phq2.m.

This script reproduces the MATLAB analysis flow using pandas/numpy/scipy/
matplotlib.  Run it from the directory containing this file, or pass explicit
CSV/output paths with command-line flags.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


RNG = np.random.default_rng(1)


@dataclass
class Config:
    spike_csv: Path = Path("/Users/edwardyao/Downloads/spike_counts.csv")
    report_csv: Path = Path("/Users/edwardyao/Downloads/clinical_data_deidentified.csv")
    output_dir: Path = Path("/Users/edwardyao/Downloads/spike_project_output")
    max_routine_hours: float = 4.0
    n_boot: int = 5000
    alpha: float = 0.05
    count_col: str = "count_0_46"
    dur_col: str = "Duration_sec"


NESD_LABEL = "Non-Epileptic Seizure Disorder"
BAD_TYPES = {s.lower() for s in ["Uncertain if Epilepsy", "Unknown or MRN not found", ""]}
CANONICAL3 = ["General", "Temporal", "Frontal"]
ALLOWABLE_VISITS = [
    "CONSULT VISIT",
    "ESTABLISHED PATIENT VISIT",
    "FOLLOW-UP PATIENT CLINIC",
    "NEW PATIENT CLINIC",
    "NEW PATIENT VISIT",
    "NPV MANAGEMENT DURING COVID-19",
    "NPV NEUROLOGY",
    "RETURN ANNUAL VISIT",
    "RETURN PATIENT EXTENDED",
    "RETURN PATIENT VISIT",
    "RPV MANAGEMENT DURING COVID-19",
    "TELEHEALTH VIDEO VISIT RETURN",
]

EPS_RATE = 30e-3
Y_ZERO = math.log10(EPS_RATE)
Y_LIMS = (-2, 4)
SPEARMAN_X_LIMS = (-3.5, 4)
SPEARMAN_Y_LIMS = (-1.5, 3)
LOW_TERTILE = 0.333
HIGH_TERTILE = 0.667


def read_csv_stringy(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def numeric(s: pd.Series | Sequence) -> pd.Series:
    return pd.to_numeric(pd.Series(s), errors="coerce")


def require_cols(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def assert_unique_keys(df: pd.DataFrame, pid_col: str, ses_col: str, name: str) -> None:
    n_keys = df[[pid_col, ses_col]].astype(str).drop_duplicates().shape[0]
    if n_keys != len(df):
        raise ValueError(f"{name} has duplicated (Patient,Session) keys ({len(df) - n_keys} duplicates).")


def _clean_scalar(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _json_loads_loose(s: str):
    s = _clean_scalar(s)
    if s in ("", "[]", "<missing>"):
        return []
    return json.loads(s.replace("NaN", "null"))


def json_to_string_array(s: str) -> list[str]:
    dec = _json_loads_loose(s)
    if isinstance(dec, list):
        return ["" if x is None else str(x) for x in dec]
    if isinstance(dec, (str, int, float)):
        return [str(dec)]
    raise TypeError(f"Unsupported JSON string-array type: {type(dec).__name__}")


def json_to_double_array(s: str) -> np.ndarray:
    s = _clean_scalar(s)
    if s in ("", "[]", "<missing>"):
        return np.array([], dtype=float)
    dec = json.loads(s.replace("null", "NaN").replace("NULL", "NaN"))
    if not isinstance(dec, list):
        dec = [dec]
    out = []
    for x in dec:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def json_dumps_strings(vals: Sequence[str]) -> str:
    return json.dumps([str(v) for v in vals], separators=(",", ":"))


def json_dumps_numbers(vals: Sequence[float]) -> str:
    cleaned = [None if not np.isfinite(v) else float(v) for v in vals]
    return json.dumps(cleaned, separators=(",", ":"))


def matlab_mean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def matlab_median(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def max_has_sz(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if x.size else np.nan


def local_frac_has_sz1(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    valid = (x == 0) | (x == 1)
    if not np.any(valid):
        return np.nan
    return float(np.sum(x[valid] == 1) / np.sum(valid))


def local_first_nonmissing(s: Sequence) -> str:
    for v in s:
        txt = _clean_scalar(v)
        if txt:
            return txt
    return ""


def p_label(p: float) -> str:
    if pd.isna(p):
        return "p=NaN"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.2g}"
    return f"p={p:.2f}"


def format_p_html(p: float) -> str:
    if pd.isna(p):
        return "p = NaN"
    if p < 0.001:
        return "p &lt; 0.001"
    if p < 0.01:
        return f"p = {p:.2g}"
    return f"p = {p:.2f}"


def spearman(x, y) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return np.nan, np.nan
    r, p = stats.spearmanr(x[m], y[m])
    return float(r), float(p)


def ranksum(x1, x2) -> float:
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]
    if len(x1) == 0 or len(x2) == 0:
        return np.nan
    return float(stats.mannwhitneyu(x1, x2, alternative="two-sided", method="asymptotic").pvalue)


def signrank(x, y_or_zero=0) -> float:
    x = np.asarray(x, dtype=float)
    if np.isscalar(y_or_zero):
        y = np.full_like(x, float(y_or_zero))
    else:
        y = np.asarray(y_or_zero, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return np.nan
    try:
        return float(stats.wilcoxon(x[m], y[m], zero_method="wilcox", correction=False, method="approx").pvalue)
    except ValueError:
        return np.nan


def cliff_delta(x1, x2) -> float:
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]
    if len(x1) == 0 or len(x2) == 0:
        return np.nan
    u = stats.mannwhitneyu(x1, x2, alternative="two-sided", method="asymptotic").statistic
    return float((2 * u / (len(x1) * len(x2))) - 1)


def bootstrap_median_ci(x, n_boot: int, alpha: float) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(x))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = np.median(x[RNG.integers(0, len(x), len(x))])
    return med, float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def bootstrap_spearman_ci(x, y, n_boot: int, alpha: float) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    rho_hat, _ = spearman(x, y)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, len(x), len(x))
        boot[b], _ = spearman(x[idx], y[idx])
    return rho_hat, float(np.nanpercentile(boot, 100 * alpha / 2)), float(np.nanpercentile(boot, 100 * (1 - alpha / 2)))


def filter_visit_arrays_by_type(r: pd.DataFrame, allowable_visits: Sequence[str]) -> pd.DataFrame:
    r = r.copy()
    allowable = set(allowable_visits)
    vt_all = r["visit_type"].map(_clean_scalar)
    null_only = vt_all.isin(["[null]", "null"])
    for c in ["visit_type", "visit_dates_deid", "sz_freqs", "visit_hasSz"]:
        r.loc[null_only, c] = "[]"

    total_before = 0
    total_after = 0
    for i, row in r.iterrows():
        vt_raw = _clean_scalar(row["visit_type"])
        if vt_raw in ("", "[]", "<missing>"):
            r.loc[i, ["visit_type", "visit_dates_deid", "sz_freqs", "visit_hasSz"]] = "[]"
            continue
        vt = json_to_string_array(vt_raw)
        dates = json_to_string_array(row["visit_dates_deid"])
        sz = json_to_double_array(row["sz_freqs"])
        hs = json_to_double_array(row["visit_hasSz"])
        if not (len(vt) == len(dates) == len(sz) == len(hs)):
            raise ValueError(f"Row {i + 1}: visit arrays have mismatched lengths.")
        total_before += len(vt)
        keep = np.array([x in allowable for x in vt], dtype=bool)
        if not np.any(keep):
            r.loc[i, ["visit_type", "visit_dates_deid", "sz_freqs", "visit_hasSz"]] = "[]"
            continue
        total_after += int(np.sum(keep))
        r.at[i, "visit_type"] = json_dumps_strings(np.asarray(vt, dtype=object)[keep])
        r.at[i, "visit_dates_deid"] = json_dumps_strings(np.asarray(dates, dtype=object)[keep])
        r.at[i, "sz_freqs"] = json_dumps_numbers(sz[keep])
        r.at[i, "visit_hasSz"] = json_dumps_numbers(hs[keep])

    print(f"[Visit-type filter] Total clinic visits before filter: {total_before}")
    pct = 100 * total_after / max(1, total_before)
    print(f"[Visit-type filter] Total clinic visits after filter:  {total_after} (kept {pct:.1f}%)")
    return r


def filter_phq2_arrays_to_retained_visits(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    require_cols(r, ["visit_dates_deid", "phq2_dates_deid", "phq2_scores"], "ReportTable")
    for i, row in r.iterrows():
        kept_raw = _clean_scalar(row["visit_dates_deid"])
        kept_dates = [] if kept_raw in ("", "[]", "<missing>") else json_to_string_array(kept_raw)
        phq_dates_raw = _clean_scalar(row["phq2_dates_deid"])
        if phq_dates_raw in ("", "[]", "<missing>"):
            r.at[i, "phq2_dates_deid"] = "[]"
            r.at[i, "phq2_scores"] = "[]"
            continue
        phq_dates = json_to_string_array(phq_dates_raw)
        phq_score = json_to_double_array(row["phq2_scores"])
        if len(phq_dates) != len(phq_score):
            raise ValueError(f"Row {i + 1}: phq2 arrays have mismatched lengths.")
        if not phq_dates or not kept_dates:
            r.at[i, "phq2_dates_deid"] = "[]"
            r.at[i, "phq2_scores"] = "[]"
            continue
        keep = np.array([d in set(kept_dates) for d in phq_dates], dtype=bool)
        r.at[i, "phq2_dates_deid"] = json_dumps_strings(np.asarray(phq_dates, dtype=object)[keep])
        r.at[i, "phq2_scores"] = json_dumps_numbers(phq_score[keep])
    return r


def filter_outpatient_routine(s: pd.DataFrame, r: pd.DataFrame, dur_col: str, max_routine_hours: float):
    n_r0, n_s0 = len(r), len(s)
    acq = r["acquired_on"].map(_clean_scalar).str.lower()
    out_site = acq.str.contains("spe", na=False) | acq.str.contains("radnor", na=False)
    out_class = r["report_PATIENT_CLASS"].map(_clean_scalar).str.lower().eq("outpatient")
    out_jay = r["jay_in_or_out"].map(_clean_scalar).str.lower().eq("out")

    outpt_keys = (
        r.loc[out_site | out_class | out_jay, ["patient_id", "session_number"]]
        .drop_duplicates()
        .rename(columns={"patient_id": "Patient", "session_number": "Session"})
    )
    if outpt_keys.empty:
        raise ValueError("No outpatient sessions identified by site/class/jay flags.")
    routine = numeric(s[dur_col]).to_numpy() <= max_routine_hours * 3600
    routine &= np.isfinite(numeric(s[dur_col]).to_numpy())
    routine_keys = s.loc[routine, ["Patient", "Session"]].drop_duplicates()
    keys = outpt_keys.merge(routine_keys, on=["Patient", "Session"], how="inner")
    s2 = s.merge(keys, on=["Patient", "Session"], how="inner")
    r2 = r.merge(keys, left_on=["patient_id", "session_number"], right_on=["Patient", "Session"], how="inner")
    print(
        f"[Outpatient+routine] Kept {len(s2)}/{n_s0} spike rows ({100 * len(s2) / max(1,n_s0):.1f}%), "
        f"{len(r2)}/{n_r0} report rows ({100 * len(r2) / max(1,n_r0):.1f}%)"
    )
    return s2, r2


def build_visit_level_table_r1(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for j, row in r.iterrows():
        pid = float(row["patient_id"])
        ds = _clean_scalar(row["visit_dates_deid"])
        if ds in ("", "[]"):
            continue
        dates = pd.to_datetime(json_to_string_array(ds), format="%Y-%m-%d", errors="coerce")
        v = json_to_double_array(row["sz_freqs"])
        v[~np.isfinite(v) | (v < 0)] = np.nan
        h = json_to_double_array(row["visit_hasSz"])
        h[h == 2] = np.nan
        if not (len(dates) == len(v) == len(h)):
            raise ValueError(f"Row {j + 1} (patient {pid:g}): visit arrays mismatched after filtering.")
        for d, freq, has in zip(dates, v, h):
            rows.append((pid, d, freq, has))
    pv = pd.DataFrame(rows, columns=["Patient", "VisitDate", "Freq", "HasSz"])
    if pv.empty:
        return pd.DataFrame(columns=["Patient", "VisitDate", "Freq", "HasSz", "Freq_R1"])
    vuniq = (
        pv.groupby(["Patient", "VisitDate"], dropna=False)
        .agg(Freq=("Freq", matlab_mean), HasSz=("HasSz", max_has_sz))
        .reset_index()
    )
    vuniq["Freq_R1"] = vuniq["Freq"]
    mask = ~np.isfinite(vuniq["Freq_R1"]) & (vuniq["HasSz"] == 0)
    vuniq.loc[mask, "Freq_R1"] = 0.0
    return vuniq


def build_patient_seizure_metrics(vuniq: pd.DataFrame) -> pd.DataFrame:
    return (
        vuniq.groupby("Patient", dropna=False)
        .agg(MeanSzFreq=("Freq_R1", matlab_mean), FracVisits_HasSz1=("HasSz", local_frac_has_sz1))
        .reset_index()
    )


def build_patient_phq2_metrics(r: pd.DataFrame) -> pd.DataFrame:
    require_cols(r, ["patient_id", "phq2_dates_deid", "phq2_scores"], "ReportTable")
    rows = []
    for j, row in r.iterrows():
        pid = float(row["patient_id"])
        ds = _clean_scalar(row["phq2_dates_deid"])
        if ds in ("", "[]", "<missing>"):
            continue
        dates = pd.to_datetime(json_to_string_array(ds), format="%Y-%m-%d", errors="coerce")
        scores = json_to_double_array(row["phq2_scores"])
        if len(dates) != len(scores):
            raise ValueError(f"Row {j + 1} (patient {pid:g}): PHQ2 arrays mismatched.")
        for d, score in zip(dates, scores):
            rows.append((pid, d, score))
    pv = pd.DataFrame(rows, columns=["Patient", "PHQ2Date", "PHQ2"])
    if pv.empty:
        return pd.DataFrame(columns=["Patient", "MeanPHQ2", "N_PHQ2_Visits"])
    pvuniq = pv.groupby(["Patient", "PHQ2Date"], dropna=False).agg(PHQ2=("PHQ2", matlab_mean)).reset_index()
    return (
        pvuniq.groupby("Patient", dropna=False)
        .agg(MeanPHQ2=("PHQ2", matlab_mean), N_PHQ2_Visits=("PHQ2", lambda x: int(np.sum(np.isfinite(x)))))
        .reset_index()
    )


def build_patient_typing_from_report(r: pd.DataFrame, canonical3: Sequence[str]) -> pd.DataFrame:
    rr = r.copy()
    rr["Patient"] = numeric(rr["patient_id"]).astype(float)
    out = pd.DataFrame({"Patient": sorted(rr["Patient"].dropna().unique())})
    for source, dest in [("epilepsy_type", "EpilepsyType"), ("epilepsy_specific", "EpilepsySpecific")]:
        vals = rr[["Patient", source]].copy()
        vals[source] = vals[source].map(_clean_scalar)
        vals = vals[vals[source].str.len() > 0]
        collapsed = []
        for pid, g in vals.sort_values("Patient").groupby("Patient"):
            u = sorted(set(g[source]) - {""})
            if len(u) > 1:
                raise ValueError(f"Conflicting {source} for Patient {pid:g}: {', '.join(u)}")
            collapsed.append((pid, u[0] if u else ""))
        out = out.merge(pd.DataFrame(collapsed, columns=["Patient", dest]), on="Patient", how="left")
    out[["EpilepsyType", "EpilepsySpecific"]] = out[["EpilepsyType", "EpilepsySpecific"]].fillna("")
    spec_norm = out["EpilepsySpecific"].map(_clean_scalar).str.lower()
    type_norm = out["EpilepsyType"].map(_clean_scalar).str.lower()
    e3 = np.array([""] * len(out), dtype=object)
    e3[spec_norm.str.contains("temporal", na=False).to_numpy()] = "Temporal"
    e3[spec_norm.str.contains("frontal", na=False).to_numpy()] = "Frontal"
    e3[(e3 == "") & type_norm.eq("general").to_numpy()] = "General"
    out["EpiType3"] = pd.Categorical(e3, categories=list(canonical3))
    return out


def build_filtered_view(sessions: pd.DataFrame, report: pd.DataFrame, typing_all: pd.DataFrame, sz_freq: pd.DataFrame):
    sess_keys = sessions[["Patient", "Session"]].drop_duplicates()
    report_kept = report.merge(sess_keys, left_on=["patient_id", "session_number"], right_on=["Patient", "Session"], how="inner")
    report_kept["Patient"] = numeric(report_kept["patient_id"]).astype(float)
    report_kept["Session"] = numeric(report_kept["session_number"]).astype(float)

    patients_kept = pd.DataFrame({"Patient": numeric(sessions["Patient"]).astype(float).drop_duplicates()})
    typing_filtered = typing_all.merge(patients_kept, on="Patient", how="inner")

    tmp = sessions.copy()
    tmp["Patient"] = numeric(tmp["Patient"]).astype(float)
    tmp["SpikeRate_perHour"] = numeric(tmp["SpikeRate_perHour"])
    pl = tmp.groupby("Patient").agg(MeanSpikeRate_perHour=("SpikeRate_perHour", matlab_mean)).reset_index()
    pl = pl.merge(typing_filtered[["Patient", "EpilepsyType", "EpilepsySpecific", "EpiType3"]], on="Patient", how="inner")

    etype_norm = pl["EpilepsyType"].map(_clean_scalar).str.lower()
    is_nesd = etype_norm.eq(NESD_LABEL.lower())
    is_bad = etype_norm.isin(BAD_TYPES) | etype_norm.eq("")
    is_epilepsy = ~(is_nesd | is_bad)

    session_level = sessions[["Patient", "Session", "SpikeRate_perHour"]].copy()
    session_level = session_level.rename(columns={"SpikeRate_perHour": "SpikesPerHour"})
    session_level[["Patient", "Session", "SpikesPerHour"]] = session_level[["Patient", "Session", "SpikesPerHour"]].apply(numeric)

    sz_filtered = sz_freq.merge(patients_kept, on="Patient", how="inner")
    ep_patients = pd.DataFrame({"Patient": pl.loc[is_epilepsy, "Patient"]})
    sz_epi = sz_filtered.merge(ep_patients, on="Patient", how="inner")
    cohort = pd.DataFrame({"Patient": sorted(sz_epi.loc[np.isfinite(sz_epi["MeanSzFreq"]), "Patient"].unique())})

    pl = pl.merge(cohort, on="Patient", how="inner")
    typing_filtered = typing_filtered.merge(cohort, on="Patient", how="inner")
    sessions = sessions.merge(cohort, on="Patient", how="inner")
    report_kept = report_kept.merge(cohort, on="Patient", how="inner")
    sz_epi = sz_epi.merge(cohort, on="Patient", how="inner")

    etype_norm = pl["EpilepsyType"].map(_clean_scalar).str.lower()
    is_nesd = etype_norm.eq(NESD_LABEL.lower()).to_numpy()
    is_bad = (etype_norm.isin(BAD_TYPES) | etype_norm.eq("")).to_numpy()
    is_epilepsy = ~(is_nesd | is_bad)
    if not np.all(is_epilepsy):
        raise AssertionError("Cohort restriction should leave only epilepsy patients.")
    print(f"[Cohort restriction] Using {len(cohort)} epilepsy patients with documented seizure frequency")

    all_tbl = pl.loc[is_epilepsy, ["Patient", "MeanSpikeRate_perHour"]].merge(sz_epi, on="Patient", how="inner")
    all_tbl = all_tbl[np.isfinite(all_tbl["MeanSpikeRate_perHour"]) & np.isfinite(all_tbl["MeanSzFreq"])]

    e3_str = pl["EpiType3"].astype(str)
    keep_canon = e3_str.isin(CANONICAL3)
    sz_canon = sz_epi.merge(pl.loc[is_epilepsy & keep_canon.to_numpy(), ["Patient", "EpiType3"]], on="Patient", how="inner")
    typed = pl.loc[is_epilepsy & keep_canon.to_numpy(), ["Patient", "MeanSpikeRate_perHour"]].merge(sz_canon, on="Patient", how="inner")
    typed = typed[np.isfinite(typed["MeanSpikeRate_perHour"]) & np.isfinite(typed["MeanSzFreq"]) & typed["EpiType3"].astype(str).isin(CANONICAL3)]

    canon_subset = pl.loc[is_epilepsy & keep_canon.to_numpy(), ["Patient", "EpiType3", "MeanSpikeRate_perHour"]].copy()
    canon_subset = canon_subset.rename(columns={"EpiType3": "EpiType4"})
    canon_subset["EpiType4"] = pd.Categorical(canon_subset["EpiType4"], categories=CANONICAL3)
    stats_rows = []
    for cat in CANONICAL3:
        x = canon_subset.loc[canon_subset["EpiType4"].astype(str) == cat, "MeanSpikeRate_perHour"].to_numpy(float)
        x = x[np.isfinite(x)]
        stats_rows.append((cat, len(x), matlab_median(x), np.percentile(x, 25) if len(x) else np.nan, np.percentile(x, 75) if len(x) else np.nan))
    canon_stats = pd.DataFrame(stats_rows, columns=["EpiType4", "GroupCount", "Median", "P25", "P75"])
    pairs = [("General", "Temporal"), ("General", "Frontal"), ("Temporal", "Frontal")]
    p_pair = []
    for a, b in pairs:
        xa = canon_subset.loc[canon_subset["EpiType4"].astype(str) == a, "MeanSpikeRate_perHour"]
        xb = canon_subset.loc[canon_subset["EpiType4"].astype(str) == b, "MeanSpikeRate_perHour"]
        p_pair.append(ranksum(xa, xb) if np.sum(np.isfinite(xa)) >= 3 and np.sum(np.isfinite(xb)) >= 3 else np.nan)

    return SimpleNamespace(
        SessionsForFigures=sessions,
        ReportForKeptSessions=report_kept,
        PatientTypingFiltered=typing_filtered,
        SessionLevelSpikeRates=session_level.merge(cohort, on="Patient", how="inner"),
        PatientLevelSpikeRates=pl,
        PatientSpikeSz_All=all_tbl,
        PatientSpikeSz_Typed=typed,
        IsEpilepsyMask=is_epilepsy,
        IsNESDMask=is_nesd,
        Canonical3_SubsetTable=canon_subset,
        Canonical3_Stats=canon_stats,
        Canonical3_Pairs=pairs,
        PvalsPairwise=np.asarray(p_pair, dtype=float),
        PvalsPairwiseBonf=np.minimum(np.asarray(p_pair, dtype=float) * 3, 1),
    )


def resolve_reported_spike_status(report: pd.DataFrame) -> pd.DataFrame:
    main = report["report_SPORADIC_EPILEPTIFORM_DISCHARGES"].map(_clean_scalar)
    is_main_present = main.eq("present")
    is_main_absent = main.eq("absent")
    raw_f = report["jay_focal_epi"].map(_clean_scalar).str.lower()
    raw_m = report["jay_multifocal_epi"].map(_clean_scalar).str.lower()
    raw_g = report["jay_gen_epi"].map(_clean_scalar).str.lower()
    f_p, f_a = raw_f.eq("present"), raw_f.eq("absent")
    m_p, m_a = raw_m.eq("present"), raw_m.eq("absent")
    g_p, g_a = raw_g.eq("present"), raw_g.eq("absent")
    present_jay_any = f_p | m_p | g_p
    all_jay_absent = f_a & m_a & g_a
    all_jay_present = f_p & m_p & g_p
    blank_main = ~(is_main_present | is_main_absent)
    blank_jay_all = ~(f_p | f_a) & ~(m_p | m_a) & ~(g_p | g_a)
    if np.any((all_jay_absent & is_main_present) | (is_main_absent & all_jay_present)):
        raise ValueError("Discordant spike presence between main and jay_* columns.")
    rep = np.array(["unknown"] * len(report), dtype=object)
    rep[(is_main_present | present_jay_any).to_numpy()] = "present"
    rep[(all_jay_absent & blank_main).to_numpy()] = "absent"
    rep[(is_main_absent & blank_jay_all).to_numpy()] = "absent"
    return pd.DataFrame({"Patient": numeric(report["Patient"]), "Session": numeric(report["Session"]), "ReportStatus": pd.Categorical(rep, categories=["absent", "present", "unknown"])})


def to_log10_per_hour(x, eps_rate=EPS_RATE):
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x) & (x > 0), x, eps_rate)
    return np.log10(x)


def to_log10_per_month(x, eps_freq=1e-3):
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x) & (x > 0), x, eps_freq)
    return np.log10(x)


def add_y_jitter_eps(y, y_zero, y_lims, frac):
    y = np.asarray(y, dtype=float).copy()
    m = np.abs(y - y_zero) < 1e-9
    if np.any(m):
        y[m] += (RNG.random(np.sum(m)) - 0.5) * frac * (y_lims[1] - y_lims[0])
    return y


def set_log10_ticks(ax, axis: str, eps_val: float, lims: Sequence[float], max_pow: int = 6):
    decades = 10.0 ** np.arange(0, max_pow + 1)
    logs = np.log10(decades)
    keep = (logs >= lims[0]) & (logs <= lims[1])
    ticks = list(logs[keep])
    labels = [f"{int(v)}" for v in decades[keep]]
    log_eps = math.log10(eps_val)
    if lims[0] <= log_eps <= lims[1]:
        ticks.insert(0, log_eps)
        labels.insert(0, "0")
    if axis.lower() == "x":
        ax.set_xticks(ticks, labels)
    else:
        ax.set_yticks(ticks, labels)


def add_sigbar(ax, x1, x2, y, text):
    yl = ax.get_ylim()
    tick = 0.03 * (yl[1] - yl[0])
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color="k", lw=1.3)
    y_off = -0.012 * (yl[1] - yl[0]) if text in ("**", "***") else 0.003 * (yl[1] - yl[0])
    ax.text(np.mean([x1, x2]), y + y_off, text, ha="center", va="bottom", fontsize=14)


def box_swarm(ax, groups: list[np.ndarray], labels: list[str], ylabel: str, ylims=None):
    ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True, boxprops={"facecolor": "0.8", "alpha": 0.35})
    for i, vals in enumerate(groups, start=1):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        x = RNG.normal(i, 0.055, size=len(vals))
        ax.scatter(x, vals, s=18, alpha=0.22)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if ylims:
        ax.set_ylim(*ylims)


def add_median_ci_overlay(ax, xpos, med, lo, hi, eps_floor, per_month=False):
    conv = to_log10_per_month if per_month else to_log10_per_hour
    y_med = conv([med], eps_floor)[0]
    y_lo = conv([lo], eps_floor)[0]
    y_hi = conv([hi], eps_floor)[0]
    ax.plot([xpos, xpos], [y_lo, y_hi], color="k", lw=3)
    ax.plot([xpos], [y_med], "ko", ms=5)


def make_fig1_controls(views, out_path: Path, n_boot: int, alpha: float):
    session_level = views.SessionLevelSpikeRates
    report = views.ReportForKeptSessions
    rs = resolve_reported_spike_status(report)
    join_a = session_level[["Patient", "Session", "SpikesPerHour"]].merge(rs, on=["Patient", "Session"], how="inner")
    x_abs = join_a.loc[join_a["ReportStatus"].astype(str) == "absent", "SpikesPerHour"].to_numpy(float)
    x_pre = join_a.loc[join_a["ReportStatus"].astype(str) == "present", "SpikesPerHour"].to_numpy(float)
    p_a = ranksum(x_abs, x_pre)
    effect_a = cliff_delta(x_pre, x_abs)
    med_abs, lo_abs, hi_abs = bootstrap_median_ci(x_abs, n_boot, alpha)
    med_pre, lo_pre, hi_pre = bootstrap_median_ci(x_pre, n_boot, alpha)

    sub = views.PatientSpikeSz_Typed[["EpiType3", "MeanSpikeRate_perHour"]].copy()
    groups_sub = [sub.loc[sub["EpiType3"].astype(str) == c, "MeanSpikeRate_perHour"].to_numpy(float) for c in CANONICAL3]
    finite_groups = [g[np.isfinite(g)] for g in groups_sub if np.sum(np.isfinite(g)) > 0]
    kw = stats.kruskal(*finite_groups) if len(finite_groups) >= 2 else SimpleNamespace(pvalue=np.nan, statistic=np.nan)
    p_kw = float(kw.pvalue)
    all_vals = np.concatenate(finite_groups) if finite_groups else np.array([])
    ss_total = float(np.sum((all_vals - np.nanmean(all_vals)) ** 2)) if all_vals.size else np.nan
    grand = np.nanmean(all_vals) if all_vals.size else np.nan
    ss_group = sum(len(g) * (np.nanmean(g) - grand) ** 2 for g in finite_groups) if all_vals.size else np.nan
    eta2 = ss_group / ss_total if ss_total and np.isfinite(ss_total) else np.nan

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 5.2), constrained_layout=True)
    y_a = add_y_jitter_eps(to_log10_per_hour(np.r_[x_abs, x_pre], EPS_RATE), Y_ZERO, Y_LIMS, 0.02)
    box_swarm(axs[0], [y_a[: len(x_abs)], y_a[len(x_abs) :]], [f"Absent (N={np.sum(np.isfinite(x_abs))})", f"Present (N={np.sum(np.isfinite(x_pre))})"], "Spikes/hour (log scale)", Y_LIMS)
    axs[0].axhline(Y_ZERO, color="0.4", ls=":", lw=1.2)
    set_log10_ticks(axs[0], "y", EPS_RATE, Y_LIMS)
    axs[0].set_title("A. Reported presence or absence of spikes")
    add_median_ci_overlay(axs[0], 1, med_abs, lo_abs, hi_abs, EPS_RATE)
    add_median_ci_overlay(axs[0], 2, med_pre, lo_pre, hi_pre, EPS_RATE)
    add_sigbar(axs[0], 1, 2, Y_LIMS[1] - 0.08 * (Y_LIMS[1] - Y_LIMS[0]), p_label(p_a))

    y_groups = [add_y_jitter_eps(to_log10_per_hour(g, EPS_RATE), Y_ZERO, Y_LIMS, 0.02) for g in groups_sub]
    labels = []
    for c, g in zip(CANONICAL3, groups_sub):
        labels.append(f"{c} (N={np.sum(np.isfinite(g))})")
    box_swarm(axs[1], y_groups, labels, "Spikes/hour (log scale)", Y_LIMS)
    axs[1].axhline(Y_ZERO, color="0.4", ls=":", lw=1.2)
    set_log10_ticks(axs[1], "y", EPS_RATE, Y_LIMS)
    axs[1].set_title("B. Epilepsy subtype")
    for k, g in enumerate(groups_sub, start=1):
        med, lo, hi = bootstrap_median_ci(g, n_boot, alpha)
        add_median_ci_overlay(axs[1], k, med, lo, hi, EPS_RATE)
    for i, ((a, b), pval) in enumerate(zip(views.Canonical3_Pairs, views.PvalsPairwiseBonf)):
        lab = "***" if pval < 1e-3 else "**" if pval < 1e-2 else "*" if pval < 5e-2 else "ns"
        add_sigbar(axs[1], CANONICAL3.index(a) + 1, CANONICAL3.index(b) + 1, Y_LIMS[1] - (0.05 + 0.08 * i) * (Y_LIMS[1] - Y_LIMS[0]), lab)
    for ax in axs:
        ax.tick_params(axis="x", labelrotation=20)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    sub_ci_rows = []
    for c, g in zip(CANONICAL3, groups_sub):
        med, lo, hi = bootstrap_median_ci(g, n_boot, alpha)
        sub_ci_rows.append((c, med, lo, hi))
    return SimpleNamespace(
        p_rankSum_A=p_a,
        effectA_cliff=effect_a,
        m_pre=med_pre,
        lo_pre=lo_pre,
        hi_pre=hi_pre,
        m_abs=med_abs,
        lo_abs=lo_abs,
        hi_abs=hi_abs,
        p_kw_C=p_kw,
        eta2_kw_C=eta2,
        p_pair_bonf=views.PvalsPairwiseBonf,
        SubtypeStatsTable=pd.DataFrame(sub_ci_rows, columns=["Group", "Median", "CI_lo", "CI_hi"]),
    )


def spearman_plotting_function(all_tbl, typed_tbl, fig_out: Path, freq_field: str, label_suffix: str, non_zero_only: bool, n_boot=5000, alpha=0.05):
    x_all = all_tbl["MeanSpikeRate_perHour"].to_numpy(float)
    y_all = all_tbl[freq_field].to_numpy(float)
    mask_all = np.isfinite(x_all) & np.isfinite(y_all)
    if non_zero_only:
        mask_all &= x_all > 0
    n_all = int(np.sum(mask_all))
    rs_all, p_all = spearman(x_all[mask_all], y_all[mask_all]) if n_all >= 3 else (np.nan, np.nan)
    _, rho_lo, rho_hi = bootstrap_spearman_ci(x_all[mask_all], y_all[mask_all], n_boot, alpha) if n_all >= 3 else (np.nan, np.nan, np.nan)

    rows = []
    ci_rows = []
    for g in CANONICAL3:
        m = typed_tbl["EpiType3"].astype(str).eq(g).to_numpy()
        x = typed_tbl.loc[m, "MeanSpikeRate_perHour"].to_numpy(float)
        y = typed_tbl.loc[m, freq_field].to_numpy(float)
        keep = np.isfinite(x) & np.isfinite(y)
        if non_zero_only:
            keep &= x > 0
        n = int(np.sum(keep))
        if n >= 3:
            rs, p = spearman(x[keep], y[keep])
            rho, lo, hi = bootstrap_spearman_ci(x[keep], y[keep], n_boot, alpha)
        else:
            rs = p = rho = lo = hi = np.nan
        rows.append((g, n, rs, p))
        ci_rows.append((g, rho, lo, hi))
    spearman_results = pd.DataFrame(rows, columns=["Group", "N", "Spearman_r", "p_raw"])
    spearman_results["p_bonf"] = np.minimum(spearman_results["p_raw"] * len(spearman_results), 1)
    subtype_ci = pd.DataFrame(ci_rows, columns=["Group", "rho", "ci_lo", "ci_hi"])

    x_used, y_used = x_all[mask_all], y_all[mask_all]
    minpos_rate = np.nanmin(x_used[x_used > 0]) if np.any(x_used > 0) else 1e-6
    minpos_sz = np.nanmin(y_used[y_used > 0]) if np.any(y_used > 0) else 1e-6
    eps_rate, eps_sz = 0.5 * minpos_rate, 0.5 * minpos_sz
    x_zero, y_zero = math.log10(eps_sz), math.log10(eps_rate)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axs.ravel()
    colors = {"All": "0.45", "Frontal": "#edb021", "Temporal": "#d9541a", "General": "#0072bd"}
    panels = [("All", all_tbl.loc[mask_all].copy(), "A. All epilepsy"), ("Frontal", typed_tbl[typed_tbl["EpiType3"].astype(str) == "Frontal"], "B. Frontal"), ("Temporal", typed_tbl[typed_tbl["EpiType3"].astype(str) == "Temporal"], "C. Temporal"), ("General", typed_tbl[typed_tbl["EpiType3"].astype(str) == "General"], "D. General")]
    for ax, (g, df, title) in zip(axes, panels):
        if g != "All":
            keep = np.isfinite(df["MeanSpikeRate_perHour"].astype(float)) & np.isfinite(df[freq_field].astype(float))
            if non_zero_only:
                keep &= df["MeanSpikeRate_perHour"].astype(float) > 0
            df = df.loc[keep]
        if df.empty:
            ax.axis("off")
            continue
        xr = df[freq_field].to_numpy(float)
        yr = df["MeanSpikeRate_perHour"].to_numpy(float)
        logx = np.log10(xr + (xr <= 0) * eps_sz)
        logy = np.log10(yr + (yr <= 0) * eps_rate)
        zx, zy = xr == 0, yr == 0
        nz = ~(zx | zy)
        ax.axvline(x_zero, color="0.4", ls=":", lw=1.2)
        ax.axhline(y_zero, color="0.4", ls=":", lw=1.2)
        ax.scatter(logx[nz], logy[nz], s=18, color=colors.get(g, "0.45"), alpha=0.3)
        ax.plot(logx[~nz], logy[~nz], "*", color=colors.get(g, "0.45"), ms=8)
        if np.sum(np.isfinite(logx) & np.isfinite(logy)) >= 3:
            b = np.polyfit(logx[np.isfinite(logx) & np.isfinite(logy)], logy[np.isfinite(logx) & np.isfinite(logy)], 1)
            xg = np.linspace(SPEARMAN_X_LIMS[0], SPEARMAN_X_LIMS[1], 250)
            ax.plot(xg, b[0] * xg + b[1], color="k" if g == "All" else colors[g], lw=2)
        ax.set_xlim(*SPEARMAN_X_LIMS)
        ax.set_ylim(*SPEARMAN_Y_LIMS)
        ax.set_xlabel("Seizures per month (log scale)")
        ax.set_ylabel("Spikes per hour (log scale)")
        set_log10_ticks(ax, "x", eps_sz, SPEARMAN_X_LIMS)
        set_log10_ticks(ax, "y", eps_rate, SPEARMAN_Y_LIMS)
        if g == "All":
            txt = f"rho={rs_all:.2f} [{rho_lo:.2f}-{rho_hi:.2f}], {p_label(p_all)}"
        else:
            row = spearman_results[spearman_results["Group"] == g].iloc[0]
            ci = subtype_ci[subtype_ci["Group"] == g].iloc[0]
            txt = f"rho={row.Spearman_r:.2f} [{ci.ci_lo:.2f}-{ci.ci_hi:.2f}], p_bonf{p_label(row.p_bonf)[1:]}"
        ax.set_title(f"{title}{label_suffix} (N={len(df)})", fontweight="bold")
        ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontweight="bold")
        labs = [lab.get_text() for lab in ax.get_xticklabels()]
        if labs:
            labs[-1] = ""
            ax.set_xticklabels(labs)
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=300)
    plt.close(fig)
    print(f"Saved Spearman figure: {fig_out}")
    return spearman_results, rs_all, p_all, n_all, rho_lo, rho_hi, subtype_ci


def make_figs2_sz_by_reported_spikes(views, sz_freq, out_path: Path, n_boot: int, alpha: float):
    rs = resolve_reported_spike_status(views.ReportForKeptSessions)
    rpt = rs.groupby("Patient").agg(
        HasPresent=("ReportStatus", lambda x: np.any(x.astype(str) == "present")),
        HasAbsent=("ReportStatus", lambda x: np.any(x.astype(str) == "absent")),
    ).reset_index()
    s2 = sz_freq.merge(rpt, on="Patient", how="inner")
    ep_patients = pd.DataFrame({"Patient": views.PatientLevelSpikeRates.loc[views.IsEpilepsyMask, "Patient"]})
    s2 = s2.merge(ep_patients, on="Patient", how="inner")
    freq_abs = s2.loc[s2["HasAbsent"] & ~s2["HasPresent"], "MeanSzFreq"].to_numpy(float)
    freq_pre = s2.loc[s2["HasPresent"], "MeanSzFreq"].to_numpy(float)
    freq_abs = freq_abs[np.isfinite(freq_abs)]
    freq_pre = freq_pre[np.isfinite(freq_pre)]
    p = ranksum(freq_abs, freq_pre)
    eps_freq = 1e-3
    y0 = math.log10(eps_freq)
    y_abs = add_y_jitter_eps(to_log10_per_month(freq_abs, eps_freq), y0, SPEARMAN_X_LIMS, 0.02)
    y_pre = add_y_jitter_eps(to_log10_per_month(freq_pre, eps_freq), y0, SPEARMAN_X_LIMS, 0.02)
    med1, lo1, hi1 = bootstrap_median_ci(freq_abs, n_boot, alpha)
    med2, lo2, hi2 = bootstrap_median_ci(freq_pre, n_boot, alpha)
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    box_swarm(ax, [y_abs, y_pre], [f"All EEGs: no spikes (N={len(freq_abs)})", f">=1 EEG: spikes present (N={len(freq_pre)})"], "Seizures/month (log scale)", SPEARMAN_X_LIMS)
    ax.axhline(y0, color="0.4", ls=":", lw=1.2)
    set_log10_ticks(ax, "y", eps_freq, SPEARMAN_X_LIMS)
    add_median_ci_overlay(ax, 1, med1, lo1, hi1, eps_freq, per_month=True)
    add_median_ci_overlay(ax, 2, med2, lo2, hi2, eps_freq, per_month=True)
    ax.set_title("Mean seizure frequency by reported spikes across EEGs")
    ax.tick_params(axis="x", labelrotation=20)
    add_sigbar(ax, 1, 2, max(np.r_[y_abs, y_pre]) + 0.35, p_label(p))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(
        "\nThe median [95% CI] seizure frequency was "
        f"{med1:.2f} [{lo1:.2f}-{hi1:.2f}] for patients whose EEGs had no reported spikes "
        f" and {med2:.2f} [{lo2:.2f}-{hi2:.2f}] for patients whose EEGs had reported spikes "
        f"({p_label(p)}, Cliff's d = {cliff_delta(freq_abs, freq_pre):.2f})."
    )


def compute_visit_eeg_gaps(vuniq, report):
    eeg_dt = pd.to_datetime(report["start_time_deid"].map(_clean_scalar), format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    eeg = pd.DataFrame({"Patient": numeric(report["Patient"]), "EEG_Date": eeg_dt}).dropna()
    gap = vuniq[["Patient", "VisitDate"]].copy()
    mins = []
    for _, row in gap.iterrows():
        dates = eeg.loc[eeg["Patient"] == row["Patient"], "EEG_Date"]
        if dates.empty or pd.isna(row["VisitDate"]):
            mins.append(np.nan)
        else:
            mins.append(float(np.min(np.abs((dates - row["VisitDate"]).dt.total_seconds() / 86400))))
    gap["MinAbsGap_days"] = mins
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
    vn = v_near.merge(pd.DataFrame({"Patient": base_patients}), on="Patient")
    vf = v_far.merge(pd.DataFrame({"Patient": base_patients}), on="Patient")
    vn = vn[np.isfinite(vn["Freq_R1"])]
    vf = vf[np.isfinite(vf["Freq_R1"])]
    p_both = np.intersect1d(vn["Patient"].unique(), vf["Patient"].unique())
    print(f"[Near/Far eligibility] Patients with >=1 short-gap AND >=1 long-gap visit (documented seizure freq): {len(p_both)}/{len(base_patients)} ({100 * len(p_both) / max(1,len(base_patients)):.1f}%)")
    sz_near = build_patient_seizure_metrics(v_near)[["Patient", "MeanSzFreq"]].rename(columns={"MeanSzFreq": "MeanSzFreq_near"})
    sz_far = build_patient_seizure_metrics(v_far)[["Patient", "MeanSzFreq"]].rename(columns={"MeanSzFreq": "MeanSzFreq_far"})
    j = spike_tbl.merge(sz_near, on="Patient").merge(sz_far, on="Patient")
    j = j[np.isfinite(j["MeanSpikeRate_perHour"]) & np.isfinite(j["MeanSzFreq_near"]) & np.isfinite(j["MeanSzFreq_far"])]
    if len(j) < 3:
        raise ValueError(f"Not enough patients with BOTH near and far seizure metrics (n={len(j)}).")
    x = j["MeanSpikeRate_perHour"].to_numpy(float)
    yn = j["MeanSzFreq_near"].to_numpy(float)
    yf = j["MeanSzFreq_far"].to_numpy(float)
    rho_near, _ = spearman(x, yn)
    rho_far, _ = spearman(x, yf)
    delta_obs = rho_near - rho_far
    delta = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, len(j), len(j))
        rn, _ = spearman(x[idx], yn[idx])
        rf, _ = spearman(x[idx], yf[idx])
        delta[b] = rn - rf
    ci_lo, ci_hi = np.nanpercentile(delta, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    delta_med = np.nanmedian(delta)
    p_one = np.nanmean(delta <= 0)
    p_two = 2 * min(np.nanmean(delta <= 0), np.nanmean(delta >= 0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.8), constrained_layout=True)
    ax1.hist(gaps, bins=60)
    ax1.axvline(near_days, color="k", ls="--", lw=2)
    ax1.axvline(far_days, color="k", ls="--", lw=2)
    ax1.set_xlabel("|Visit - EEG| gap (days)")
    ax1.set_ylabel("Visit count")
    ax1.set_title("A. Visit-EEG gap distribution with lower and upper third cutoffs")
    ax2.hist(delta, bins=40)
    ax2.axvline(0, color="k", ls="--", lw=2)
    ax2.axvline(delta_med, color="k", lw=2)
    max_abs = np.nanmax(np.abs(delta))
    max_abs = max_abs if np.isfinite(max_abs) and max_abs else 1e-3
    ax2.set_xlim(-1.08 * max_abs, 1.08 * max_abs)
    ax2.set_xlabel("Delta rho = rho_short gap - rho_long gap")
    ax2.set_ylabel("Bootstrap count")
    ax2.set_title(f"B. Distribution of differences in spike-seizure correlation\nbetween short and long visit-EEG gaps\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}], p = {p_one:.3g}")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(
        f"\nFig S3 analysis:\nN patients: {len(j)}\nMedian rho short gap: {rho_near:.2f}\n"
        f"Median rho long gap: {rho_far:.2f}\nMedian [95% CI] difference in rho: {delta_obs:.3f} [{ci_lo:.2f}-{ci_hi:.2f}]\np = {p_one:.4f}."
    )
    return SimpleNamespace(nPatients=len(j), nearQ=near_q, farQ=far_q, nearDays=near_days, farDays=far_days, rho_near=rho_near, rho_far=rho_far, delta_obs=delta_obs, delta_boot=delta, delta_median=delta_med, delta_ci_lo=ci_lo, delta_ci_hi=ci_hi, p_one_sided=p_one, p_two_sided=p_two, tableUsed=j, gapsUsed=gaps)


def build_table1_flat(views, sz_freq, vuniq, n_boot, alpha):
    rk, pl = views.ReportForKeptSessions, views.PatientLevelSpikeRates
    all_patients = pl["Patient"].to_numpy(float)
    n_total = len(all_patients)
    birth = pd.to_datetime(rk["deid_birth_date"].map(_clean_scalar).replace({"null": "", "[null]": ""}), format="%Y-%m-%d", errors="coerce")
    age = (pd.Timestamp(2000, 1, 1) - birth).dt.days / 365.25
    age_tbl = pd.DataFrame({"Patient": rk["Patient"], "Age": age}).groupby("Patient").agg(AgeFirst=("Age", lambda x: np.nanmin(x) if np.any(np.isfinite(x)) else np.nan)).reset_index()
    age_vec = pd.DataFrame({"Patient": all_patients}).merge(age_tbl, on="Patient", how="left")["AgeFirst"].to_numpy(float)
    sex_tbl = pd.DataFrame({"Patient": rk["Patient"], "Sex": rk["nlp_gender"].map(_clean_scalar).str.upper()}).groupby("Patient").agg(SexCode=("Sex", local_first_nonmissing)).reset_index()
    sex = pd.DataFrame({"Patient": all_patients}).merge(sex_tbl, on="Patient", how="left")["SexCode"].fillna("")
    n_f, n_m = int(np.sum(sex == "F")), int(np.sum(sex == "M"))
    n_u = n_total - n_f - n_m
    n_epi, n_pnes = int(np.sum(views.IsEpilepsyMask)), int(np.sum(views.IsNESDMask))
    e3 = pl["EpiType3"].astype(str)
    espec = pl["EpilepsySpecific"].map(_clean_scalar)
    is_temp = e3.eq("Temporal") & views.IsEpilepsyMask
    is_front = e3.eq("Frontal") & views.IsEpilepsyMask
    is_gen = e3.eq("General") & views.IsEpilepsyMask
    is_canon = is_temp | is_front | is_gen
    is_unknown = views.IsEpilepsyMask & ~is_canon & (espec.eq("") | espec.isin(["Unclassified or Unspecified", "Unknown or MRN not found"]))
    is_other = views.IsEpilepsyMask & ~is_canon & ~is_unknown
    visits = vuniq[["Patient", "VisitDate"]].groupby("Patient")["VisitDate"].nunique().reset_index(name="NumVisits")
    vis_vec = pd.DataFrame({"Patient": all_patients}).merge(visits, on="Patient", how="left")["NumVisits"].to_numpy(float)
    eeg = views.SessionsForFigures.groupby("Patient")["Session"].nunique().reset_index(name="NumEEG")
    eeg_vec = pd.DataFrame({"Patient": all_patients}).merge(eeg, on="Patient", how="left")["NumEEG"].to_numpy(float)
    sf_vec = pd.DataFrame({"Patient": all_patients}).merge(sz_freq, on="Patient", how="left")["MeanSzFreq"].to_numpy(float)
    sf_vec = sf_vec[np.isfinite(sf_vec)]
    sr_vec = pl["MeanSpikeRate_perHour"].to_numpy(float)
    sr_vec = sr_vec[np.isfinite(sr_vec)]
    _, sf_lo, sf_hi = bootstrap_median_ci(sf_vec, n_boot, alpha)
    _, sr_lo, sr_hi = bootstrap_median_ci(sr_vec, n_boot, alpha)
    rs_all = resolve_reported_spike_status(views.ReportForKeptSessions)
    stat = rs_all["ReportStatus"].astype(str)
    n_rep_pre, n_rep_abs, n_rep_unk = int(np.sum(stat == "present")), int(np.sum(stat == "absent")), int(np.sum(stat == "unknown"))
    n_eegs_all = len(rs_all)

    def pct(n, den):
        return 100 * n / max(1, den)

    rows = [
        ("Total N patients with >=1 outpatient routine EEG", f"{n_total:d}"),
        ("Age at first visit (years)", f"{np.nanmedian(age_vec):.1f} ({np.nanpercentile(age_vec,25):.1f}-{np.nanpercentile(age_vec,75):.1f})"),
        ("Sex", ""),
        ("    Women", f"{n_f:d} ({pct(n_f,n_total):.1f}%)"),
        ("    Men", f"{n_m:d} ({pct(n_m,n_total):.1f}%)"),
        ("    Unknown/Other", f"{n_u:d} ({pct(n_u,n_total):.1f}%)"),
        ("Epilepsy subtype", ""),
        ("    Temporal lobe", f"{int(is_temp.sum()):d} ({pct(int(is_temp.sum()),n_epi):.1f}%)"),
        ("    Frontal lobe", f"{int(is_front.sum()):d} ({pct(int(is_front.sum()),n_epi):.1f}%)"),
        ("    Generalized", f"{int(is_gen.sum()):d} ({pct(int(is_gen.sum()),n_epi):.1f}%)"),
        ("    Other", f"{int(is_other.sum()):d} ({pct(int(is_other.sum()),n_epi):.1f}%)"),
        ("    Unknown", f"{int(is_unknown.sum()):d} ({pct(int(is_unknown.sum()),n_epi):.1f}%)"),
        ("Number of clinic visits", f"{np.nanmedian(vis_vec):.1f} ({np.nanpercentile(vis_vec,25):.1f}-{np.nanpercentile(vis_vec,75):.1f})"),
        ("Number of EEGs", f"{np.nanmedian(eeg_vec):.1f} ({np.nanpercentile(eeg_vec,25):.1f}-{np.nanpercentile(eeg_vec,75):.1f})"),
        ("Mean seizure frequency (seizures/month)", f"{np.nanmedian(sf_vec):.2f} ({np.nanpercentile(sf_vec,25):.2f}-{np.nanpercentile(sf_vec,75):.2f}); median CI [{sf_lo:.2f}-{sf_hi:.2f}]"),
        ("Mean spike rate (spikes/hour)", f"{np.nanmedian(sr_vec):.2f} ({np.nanpercentile(sr_vec,25):.2f}-{np.nanpercentile(sr_vec,75):.2f}); median CI [{sr_lo:.2f}-{sr_hi:.2f}]"),
        ("Reported spikes", ""),
        ("    Present", f"{n_rep_pre:d} ({pct(n_rep_pre,n_eegs_all):.1f}%)"),
        ("    Absent", f"{n_rep_abs:d} ({pct(n_rep_abs,n_eegs_all):.1f}%)"),
        ("    Unknown", f"{n_rep_unk:d} ({pct(n_rep_unk,n_eegs_all):.1f}%)"),
    ]
    return pd.DataFrame(rows, columns=["Variable", "Statistic"])


def write_results_html(out_path, views, sz_freq, fig1stats, spearman_results, rs_all, p_all, n_all, rho_lo, rho_hi, subtype_ci, report):
    pl = views.PatientLevelSpikeRates
    n_total, n_eegs_all, n_epi = len(pl), len(report), int(np.sum(views.IsEpilepsyMask))
    sf = pd.DataFrame({"Patient": pl["Patient"]}).merge(sz_freq, on="Patient", how="left")["MeanSzFreq"].to_numpy(float)
    sf = sf[np.isfinite(sf)]
    sr = pl["MeanSpikeRate_perHour"].to_numpy(float)
    sr = sr[np.isfinite(sr)]
    sf_med, sf_lo, sf_hi = bootstrap_median_ci(sf, 5000, 0.05)
    sr_med, sr_lo, sr_hi = bootstrap_median_ci(sr, 5000, 0.05)
    tsub = fig1stats.SubtypeStatsTable.set_index("Group")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write('<html><head><meta charset="UTF-8"><title>Results</title></head><body>\n')
        f.write("<h2>Cohort summary</h2>\n")
        f.write(f"<p>We included {n_total} patients ({n_eegs_all} EEGs). Median [95% CI] monthly seizure frequency was {sf_med:.2f} [{sf_lo:.2f}-{sf_hi:.2f}], and median [95% CI] spikes/hour across EEGs was {sr_med:.2f} [{sr_lo:.2f}-{sr_hi:.2f}] (Table 1).</p>")
        f.write("<h2>Spike rates by patient groups</h2>\n")
        f.write(f"<p>Automatically detected spike rates were higher in EEGs with clinically-reported spikes (median spike rate {fig1stats.m_pre:.2f} [95% CI {fig1stats.lo_pre:.2f}-{fig1stats.hi_pre:.2f}] spikes/hour) than in EEGs without reported spikes ({fig1stats.m_abs:.2f} [{fig1stats.lo_abs:.2f}-{fig1stats.hi_abs:.2f}] spikes/hour) ({format_p_html(fig1stats.p_rankSum_A)}, Cliff's &delta; = {fig1stats.effectA_cliff:.2f}; Fig. 1A). ")
        f.write(f"Spike rates differed across epilepsy subtypes (Kruskal-Wallis {format_p_html(fig1stats.p_kw_C)}, eta^2 ~= {fig1stats.eta2_kw_C:.3f}), with higher rates in generalized epilepsy than temporal or frontal lobe epilepsy (Fig. 1B).</p>")
        f.write("<h2>Relationship between spike rate and seizure frequency</h2>\n")
        f.write(f"<p>Spike rate and seizure frequency were positively correlated (N = {n_all}, rho = {rs_all:.2f} [95% CI {rho_lo:.2f}-{rho_hi:.2f}], {format_p_html(p_all)}). ")
        for group in CANONICAL3:
            row = spearman_results[spearman_results["Group"] == group].iloc[0]
            ci = subtype_ci[subtype_ci["Group"] == group].iloc[0]
            f.write(f"{html.escape(group)} epilepsy: N = {int(row.N)}, rho = {row.Spearman_r:.2f} [{ci.ci_lo:.2f}-{ci.ci_hi:.2f}], Bonferroni-adjusted {format_p_html(row.p_bonf)}. ")
        f.write("Results were similar when restricting analyses to patients with detectable spikes. Patients with spikes clinically-reported on at least one EEG also had higher mean seizure frequencies than those without spikes. Spike-seizure correlations were stronger when clinic visits occurred closer in time to EEG acquisition.</p>")
        f.write("</body></html>\n")


def plot_spike_phq2_correlation(patient_spike_phq2, out_path: Path, n_boot, alpha):
    x = patient_spike_phq2["MeanPHQ2"].to_numpy(float)
    y = patient_spike_phq2["MeanSpikeRate_perHour"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        raise ValueError(f"Not enough patients with PHQ2 data (n={len(x)}).")
    rho, p = spearman(x, y)
    _, lo, hi = bootstrap_spearman_ci(x, y, n_boot, alpha)
    ylog = np.log10(np.maximum(y, EPS_RATE))
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    ax.scatter(x, ylog, s=24, alpha=0.3)
    b = np.polyfit(x, ylog, 1)
    xg = np.linspace(np.min(x), np.max(x), 200)
    ax.plot(xg, b[0] * xg + b[1], color="k", lw=2)
    ax.set_ylabel("Spikes/hour (log scale)")
    ax.set_xlabel("Mean PHQ-2 score")
    set_log10_ticks(ax, "y", EPS_RATE, Y_LIMS)
    ax.set_ylim(*Y_LIMS)
    ax.set_title(f"Spike rate vs mean PHQ-2 (N={len(x)})")
    ax.text(0.98, 0.95, f"rho={rho:.2f} [{lo:.2f}-{hi:.2f}], {p_label(p)}", transform=ax.transAxes, ha="right", va="top", fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return rho, p, len(x), lo, hi


def plot_phq2_seizure_corr(patient_sz_phq2, out_path: Path):
    x = patient_sz_phq2["MeanPHQ2"].to_numpy(float)
    y = patient_sz_phq2["MeanSzFreq"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    eps_sz = 1e-3
    ylog = np.log10(np.maximum(y, eps_sz))
    fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    ax.scatter(x, ylog, s=24, alpha=0.3)
    if len(x) >= 2:
        b = np.polyfit(x, ylog, 1)
        xg = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xg, b[0] * xg + b[1], color="k", lw=2)
    ax.set_xlabel("Mean PHQ-2 score")
    ax.set_ylabel("Seizures/month (log scale)")
    set_log10_ticks(ax, "y", eps_sz, (-3, 4))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_pipeline(cfg: Config) -> None:
    spike = read_csv_stringy(cfg.spike_csv)
    require_cols(spike, ["Patient", "Session", cfg.count_col, cfg.dur_col], "SpikeSummaryTable")
    spike[cfg.count_col] = numeric(spike[cfg.count_col])
    spike[cfg.dur_col] = numeric(spike[cfg.dur_col])
    spike["SpikeRate_perHour"] = spike[cfg.count_col] / spike[cfg.dur_col] * 3600

    report = read_csv_stringy(cfg.report_csv)
    require_cols(
        report,
        [
            "patient_id",
            "session_number",
            "acquired_on",
            "report_PATIENT_CLASS",
            "jay_in_or_out",
            "visit_type",
            "visit_dates_deid",
            "sz_freqs",
            "visit_hasSz",
            "epilepsy_type",
            "epilepsy_specific",
            "nlp_gender",
            "deid_birth_date",
            "start_time_deid",
            "report_SPORADIC_EPILEPTIFORM_DISCHARGES",
            "jay_focal_epi",
            "jay_multifocal_epi",
            "jay_gen_epi",
            "phq2_dates_deid",
            "phq2_scores",
        ],
        "ReportTable",
    )

    report = filter_visit_arrays_by_type(report, ALLOWABLE_VISITS)
    report = filter_phq2_arrays_to_retained_visits(report)
    spike, report = filter_outpatient_routine(spike, report, cfg.dur_col, cfg.max_routine_hours)
    assert_unique_keys(spike, "Patient", "Session", "SpikeSummaryTable")
    assert_unique_keys(report, "patient_id", "session_number", "ReportTable")

    vuniq = build_visit_level_table_r1(report)
    typing = build_patient_typing_from_report(report, CANONICAL3)
    sz_freq = build_patient_seizure_metrics(vuniq)
    phq2 = build_patient_phq2_metrics(report)
    views = build_filtered_view(spike, report, typing, sz_freq)

    patient_sz_phq2 = sz_freq[["Patient", "MeanSzFreq"]].merge(phq2[["Patient", "MeanPHQ2"]], on="Patient", how="inner")
    patient_sz_phq2 = patient_sz_phq2.merge(pd.DataFrame({"Patient": views.PatientLevelSpikeRates["Patient"]}), on="Patient", how="inner")
    patient_sz_phq2 = patient_sz_phq2[np.isfinite(patient_sz_phq2["MeanSzFreq"]) & np.isfinite(patient_sz_phq2["MeanPHQ2"])]
    if len(patient_sz_phq2) >= 3:
        rho, p = spearman(patient_sz_phq2["MeanPHQ2"], patient_sz_phq2["MeanSzFreq"])
        _, lo, hi = bootstrap_spearman_ci(patient_sz_phq2["MeanPHQ2"], patient_sz_phq2["MeanSzFreq"], cfg.n_boot, cfg.alpha)
        print(f"\nPHQ2 vs seizure frequency:\nN patients: {len(patient_sz_phq2)}\nSpearman rho = {rho:.2f} [95% CI {lo:.2f}-{hi:.2f}]\n{p_label(p)}")
    else:
        print("Warning: Not enough patients with PHQ2 + seizure frequency.")

    patient_spike_phq2 = views.PatientLevelSpikeRates[["Patient", "MeanSpikeRate_perHour", "EpiType3"]].merge(phq2, on="Patient", how="inner")
    patient_spike_phq2 = patient_spike_phq2[np.isfinite(patient_spike_phq2["MeanSpikeRate_perHour"]) & np.isfinite(patient_spike_phq2["MeanPHQ2"])]
    if len(patient_spike_phq2) >= 3:
        rho, p = spearman(patient_spike_phq2["MeanSpikeRate_perHour"], patient_spike_phq2["MeanPHQ2"])
        _, lo, hi = bootstrap_spearman_ci(patient_spike_phq2["MeanSpikeRate_perHour"], patient_spike_phq2["MeanPHQ2"], cfg.n_boot, cfg.alpha)
        print(f"\nPHQ2 analysis:\nN patients: {len(patient_spike_phq2)}\nSpearman rho = {rho:.2f} [95% CI {lo:.2f}-{hi:.2f}]\n{p_label(p)}")
        plot_spike_phq2_correlation(patient_spike_phq2, cfg.output_dir / "FigS4.png", cfg.n_boot, cfg.alpha)
    else:
        print("Warning: Not enough patients with PHQ2 data for correlation.")

    if len(patient_sz_phq2) >= 1:
        plot_phq2_seizure_corr(patient_sz_phq2, cfg.output_dir / "Fig_PHQ2_vs_sz.png")

    fig1stats = make_fig1_controls(views, cfg.output_dir / "Fig1.png", cfg.n_boot, cfg.alpha)
    print(f"Saved Fig 1: {cfg.output_dir / 'Fig1.png'}")

    sp_main = spearman_plotting_function(views.PatientSpikeSz_All, views.PatientSpikeSz_Typed, cfg.output_dir / "Fig2.png", "MeanSzFreq", "", False, cfg.n_boot, cfg.alpha)
    sp_nz = spearman_plotting_function(views.PatientSpikeSz_All, views.PatientSpikeSz_Typed, cfg.output_dir / "FigS1.png", "MeanSzFreq", " (positive spike rates only)", True, cfg.n_boot, cfg.alpha)
    print(f"Saved Fig 2:  {cfg.output_dir / 'Fig2.png'}")
    print(f"Saved Fig S1: {cfg.output_dir / 'FigS1.png'}")

    make_figs2_sz_by_reported_spikes(views, sz_freq, cfg.output_dir / "FigS2.png", cfg.n_boot, cfg.alpha)
    print(f"Saved Fig S2: {cfg.output_dir / 'FigS2.png'}")

    plot_delta_rho_histogram(views, vuniq, views.ReportForKeptSessions, LOW_TERTILE, HIGH_TERTILE, cfg.n_boot, cfg.alpha, cfg.output_dir / "FigS3.png")

    table1 = build_table1_flat(views, sz_freq, vuniq, cfg.n_boot, cfg.alpha)
    table_path = cfg.output_dir / "Table1.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table1.to_csv(table_path, index=False)
    print(f"Wrote Table 1 CSV: {table_path}")

    write_results_html(cfg.output_dir / "results_summary.html", views, sz_freq, fig1stats, *sp_main, views.ReportForKeptSessions)
    print(f"Wrote HTML summary: {cfg.output_dir / 'results_summary.html'}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Python port of spike_phq2.m")
    p.add_argument("--spike-csv", type=Path, default=Config.spike_csv)
    p.add_argument("--report-csv", type=Path, default=Config.report_csv)
    p.add_argument("--output-dir", type=Path, default=Config.output_dir)
    p.add_argument("--n-boot", type=int, default=Config.n_boot)
    p.add_argument("--max-routine-hours", type=float, default=Config.max_routine_hours)
    args = p.parse_args()
    return Config(spike_csv=args.spike_csv, report_csv=args.report_csv, output_dir=args.output_dir, n_boot=args.n_boot, max_routine_hours=args.max_routine_hours)


if __name__ == "__main__":
    run_pipeline(parse_args())
