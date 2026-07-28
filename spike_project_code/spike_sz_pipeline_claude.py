"""
spike_sz_pipeline.py

Python conversion of run_spike_sz_pipeline_clean.m
Manuscript: "Interictal Spike Rate on Routine Outpatient EEG Is Associated
With Seizure Frequency in a Large Epilepsy Cohort" by Conrad et al., 2026

Requirements:
    pip install pandas numpy scipy matplotlib statsmodels

Data files (place in ../data/):
    spike_counts.csv
    clinical_data_deidentified.csv

Usage:
    python spike_sz_pipeline.py
"""

import json
import os
import warnings
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import ranksums, kruskal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit
from typing import Any, cast

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
RNG_SEED          = 1
MAX_ROUTINE_HOURS = 4
NESD_LABEL        = "Non-Epileptic Seizure Disorder"
BAD_TYPES         = {"uncertain if epilepsy", "unknown or mrn not found", ""}
CANONICAL3        = ["General", "Temporal", "Frontal"]

EPS_RATE       = 30e-3
Y_ZERO         = np.log10(EPS_RATE)
Y_LIMS         = [-2, 4]
TITLE_Y_OFFSET = 0.02

SPEARMAN_XLIMS = [-3.5, 4]
SPEARMAN_YLIMS = [-1.5, 3]

N_BOOT   = 5000
ALPHA    = 0.05
COUNT_COL = "count_0_46"
DUR_COL   = "Duration_sec"

ALLOWABLE_VISITS = {
    "CONSULT VISIT", "ESTABLISHED PATIENT VISIT",
    "FOLLOW-UP PATIENT CLINIC", "NEW PATIENT CLINIC", "NEW PATIENT VISIT",
    "NPV MANAGEMENT DURING COVID-19", "NPV NEUROLOGY",
    "RETURN ANNUAL VISIT", "RETURN PATIENT EXTENDED", "RETURN PATIENT VISIT",
    "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN",
}

# Paths
DATA_DIR   = Path("../data")
OUTPUT_DIR = Path("../output")

SPIKE_SUMMARY_CSV = DATA_DIR / "/Users/edwardyao/Documents/PURM/data/spike_counts.csv"
REPORT_CSV        = DATA_DIR / "/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RNG_SEED)


# ─────────────────────────────────────────────
# SMALL UTILITIES
# ─────────────────────────────────────────────

def require_cols(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def json_to_string_array(s: str):
    s = str(s).strip()
    if s in ("", "[]", "<missing>", "nan", "None"):
        return []
    try:
        dec = json.loads(s)
    except Exception:
        return []
    if isinstance(dec, list):
        return [str(x) for x in dec]
    if isinstance(dec, (str, int, float)):
        return [str(dec)]
    return []


def json_to_double_array(s: str):
    s = str(s).strip()
    if s in ("", "[]", "<missing>", "nan", "None"):
        return np.array([], dtype=float)
    s = re.sub(r"null", "NaN", s, flags=re.IGNORECASE)
    try:
        dec = json.loads(s)
    except Exception:
        return np.array([], dtype=float)
    arr = np.array(dec, dtype=float).ravel()
    arr[~np.isfinite(arr) | (arr < 0)] = np.nan
    return arr


def to_log10_per_hour(x, eps=EPS_RATE):
    x = np.asarray(x, dtype=float).copy()
    x[(~np.isfinite(x)) | (x <= 0)] = eps
    return np.log10(x)


def to_log10_per_month(f, eps=1e-3):
    f = np.asarray(f, dtype=float).copy()
    f[(~np.isfinite(f)) | (f <= 0)] = eps
    return np.log10(f)


def add_y_jitter_eps(Y, y_zero, y_lims, frac=0.02):
    Y = np.array(Y, dtype=float).copy()
    mask = np.abs(Y - y_zero) < 1e-9
    if mask.any():
        amp = frac * (y_lims[1] - y_lims[0])
        Y[mask] += (np.random.rand(mask.sum()) - 0.5) * amp
    return Y


def p_label(p):
    if p is None or np.isnan(p):
        return "p=NaN"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.2g}"
    return f"p={p:.2f}"


def format_p_html(p):
    if p is None or np.isnan(p):
        return "p = NaN"
    if p < 0.001:
        return "p &lt; 0.001"
    if p < 0.01:
        s = f"{p:.2g}"
        return f"p = {s}"
    return f"p = {p:.2f}"


def cliff_delta(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]
    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        return np.nan
   
    greater = np.sum(x1[:, None] > x2[None, :]).item()
    less    = np.sum(x1[:, None] < x2[None, :]).item()

    return (greater - less) / (n1 * n2)


def bootstrap_median_ci(x, n_boot=N_BOOT, alpha=ALPHA):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    med = np.nanmedian(x)
    if n_boot == 0 or len(x) == 0:
        return med, np.nan, np.nan
    n = len(x)
    boot = np.array([np.median(x[np.random.randint(0, n, n)]) for _ in range(n_boot)])
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return med, lo, hi


def bootstrap_spearman_ci(x, y, n_boot=5000, alpha=0.05) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    rho_hat = spearman_rho(x, y)

    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        boot[i] = spearman_rho(x[idx], y[idx])

    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))

    return rho_hat, lo, hi


def set_log10_ticks(ax, which_axis, eps_val, axis_lims, max_pow=6):
    decades   = 10.0 ** np.arange(0, max_pow + 1)
    log_dec   = np.log10(decades)
    keep      = (log_dec >= axis_lims[0]) & (log_dec <= axis_lims[1])
    ticks     = list(log_dec[keep])
    labels    = [str(int(d)) for d in decades[keep]]
    log_eps   = np.log10(eps_val)
    if axis_lims[0] <= log_eps <= axis_lims[1]:
        ticks  = [log_eps] + ticks
        labels = ["0"] + labels
    if which_axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def add_sigbar(ax, x1, x2, y, ptext):
    ylim = ax.get_ylim()
    tick = 0.03 * (ylim[1] - ylim[0])
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], "k-", linewidth=1.3)
    y_off = -0.012 * (ylim[1] - ylim[0]) if ptext in ("**", "***") else 0.003 * (ylim[1] - ylim[0])
    ax.text(np.mean([x1, x2]), y + y_off, ptext, ha="center", va="bottom", fontsize=16)


def add_median_ci_overlay(ax, xpos, med_raw, lo_raw, hi_raw, eps_floor=EPS_RATE):
    y_med = to_log10_per_hour(np.atleast_1d(med_raw), eps_floor)[0]
    y_lo  = to_log10_per_hour(np.atleast_1d(lo_raw),  eps_floor)[0]
    y_hi  = to_log10_per_hour(np.atleast_1d(hi_raw),  eps_floor)[0]
    ax.plot([xpos, xpos], [y_lo, y_hi], "k-", linewidth=3)
    ax.plot(xpos, y_med, "ko", markersize=6, markerfacecolor="k")


def save_fig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    spike_df  = pd.read_csv(SPIKE_SUMMARY_CSV, dtype=str)
    report_df = pd.read_csv(REPORT_CSV,         dtype=str)

    require_cols(spike_df,  ["Patient", "Session", COUNT_COL, DUR_COL], "SpikeSummaryTable")
    require_cols(report_df, [
        "patient_id", "session_number", "acquired_on",
        "report_PATIENT_CLASS", "jay_in_or_out",
        "visit_type", "visit_dates_deid", "sz_freqs", "visit_hasSz",
        "epilepsy_type", "epilepsy_specific", "nlp_gender", "deid_birth_date", "start_time_deid",
        "report_SPORADIC_EPILEPTIFORM_DISCHARGES", "jay_focal_epi", "jay_multifocal_epi", "jay_gen_epi"
    ], "ReportTable")

    spike_df[COUNT_COL]          = pd.to_numeric(spike_df[COUNT_COL], errors="coerce")
    spike_df[DUR_COL]            = pd.to_numeric(spike_df[DUR_COL],   errors="coerce")
    spike_df["SpikeRate_perHour"] = spike_df[COUNT_COL] / spike_df[DUR_COL] * 3600

    spike_df["Patient"] = pd.to_numeric(spike_df["Patient"], errors="coerce")
    spike_df["Session"] = pd.to_numeric(spike_df["Session"], errors="coerce")

    report_df["patient_id"]     = pd.to_numeric(report_df["patient_id"],     errors="coerce")
    report_df["session_number"] = pd.to_numeric(report_df["session_number"], errors="coerce")

    return spike_df, report_df


# ─────────────────────────────────────────────
# FILTER VISIT ARRAYS BY TYPE
# ─────────────────────────────────────────────

def filter_visit_arrays_by_type(df: pd.DataFrame, allowable_visits: set) -> pd.DataFrame:
    df = df.copy()

    for col in ["visit_type", "visit_dates_deid", "sz_freqs", "visit_hasSz"]:
        mask_null = df[col].astype(str).str.strip().isin(["[null]", "null"])
        df.loc[mask_null, col] = "[]"

    total_before, total_after = 0, 0

    for i in df.index:
        vt_raw = str(df.at[i, "visit_type"]).strip()
        if vt_raw in ("", "[]", "<missing>", "nan"):
            for c in ["visit_type", "visit_dates_deid", "sz_freqs", "visit_hasSz"]:
                df.at[i, c] = "[]"
            continue

        vt     = json_to_string_array(vt_raw)
        dates  = json_to_string_array(str(df.at[i, "visit_dates_deid"]).strip())
        sz     = json_to_double_array(str(df.at[i, "sz_freqs"]).strip())
        hs     = json_to_double_array(str(df.at[i, "visit_hasSz"]).strip())

        # ─────────────────────────────────────
        # FIX: enforce alignment instead of crashing
        # ─────────────────────────────────────
        min_len = min(len(vt), len(dates), len(sz), len(hs))

        vt    = vt[:min_len]
        dates = dates[:min_len]
        sz    = sz[:min_len]
        hs    = hs[:min_len]

        total_before += len(vt)

        keep = [v.strip().upper() in allowable_visits for v in vt]

        if not any(keep):
            for c in ["visit_type", "visit_dates_deid", "sz_freqs", "visit_hasSz"]:
                df.at[i, c] = "[]"
            continue

        vt_f    = [vt[k]    for k in range(len(vt))    if keep[k]]
        dates_f = [dates[k] for k in range(len(dates)) if keep[k]]
        sz_f    = sz[keep].tolist()
        hs_f    = hs[keep].tolist()

        total_after += len(vt_f)

        df.at[i, "visit_type"]       = json.dumps(vt_f)
        df.at[i, "visit_dates_deid"] = json.dumps(dates_f)
        df.at[i, "sz_freqs"]         = json.dumps(sz_f)
        df.at[i, "visit_hasSz"]      = json.dumps(hs_f)

    print(f"[Visit-type filter] {total_before} -> {total_after} visits "
          f"(kept {100*total_after/max(1,total_before):.1f}%)")

    return df

# ─────────────────────────────────────────────
# FILTER OUTPATIENT ROUTINE
# ─────────────────────────────────────────────

def filter_outpatient_routine(spike_df, report_df, dur_col, max_hours):
    n_patients_total = report_df["patient_id"].nunique()
    n_r0 = len(report_df)
    n_s0 = len(spike_df)

    acq   = report_df["acquired_on"].str.lower().str.strip()
    cls_  = report_df["report_PATIENT_CLASS"].str.lower().str.strip()
    jay   = report_df["jay_in_or_out"].str.lower().str.strip()

    is_outpt = (
        acq.str.contains("spe", na=False) |
        acq.str.contains("radnor", na=False) |
        (cls_ == "outpatient") |
        (jay == "out")
    )
    outpt_keys = report_df.loc[is_outpt, ["patient_id", "session_number"]].drop_duplicates()
    outpt_keys = outpt_keys.rename(columns={"patient_id": "Patient", "session_number": "Session"})

    is_routine = (
        pd.to_numeric(spike_df[dur_col], errors="coerce").notna() &
        (pd.to_numeric(spike_df[dur_col], errors="coerce") <= max_hours * 3600)
    )
    routine_keys = spike_df.loc[is_routine, ["Patient", "Session"]].drop_duplicates()

    merged_keys = outpt_keys.merge(routine_keys, on=["Patient", "Session"])

    spike_df  = spike_df.merge(merged_keys, on=["Patient", "Session"])
    report_df = report_df.merge(
        merged_keys.rename(columns={"Patient": "patient_id", "Session": "session_number"}),
        on=["patient_id", "session_number"]
    )

    print(f"[Outpatient+routine] Kept {len(spike_df)}/{n_s0} spike rows, "
          f"{len(report_df)}/{n_r0} report rows")
    return spike_df, report_df, n_patients_total


# ─────────────────────────────────────────────
# BUILD VISIT-LEVEL TABLE
# ─────────────────────────────────────────────

def build_visit_level_table(report_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in report_df.iterrows():
        pid = float(row["patient_id"])
        ds  = str(row["visit_dates_deid"]).strip()
        if ds in ("[]", "", "nan"):
            continue
        try:
            dates = pd.to_datetime(json.loads(ds), format="%Y-%m-%d")
        except Exception:
            continue

        s = str(row["sz_freqs"]).strip()
        s = re.sub(r"null", "NaN", s, flags=re.IGNORECASE)
        try:
            v = np.array(json.loads(s), dtype=float).ravel()
        except Exception:
            v = np.full(len(dates), np.nan)
        v[~np.isfinite(v) | (v < 0)] = np.nan

        hs = str(row["visit_hasSz"]).strip()
        try:
            h = np.array(json.loads(hs), dtype=float).ravel()
        except Exception:
            h = np.full(len(dates), np.nan)
        h[h == 2] = np.nan

        if not (len(dates) == len(v) == len(h)):
            raise ValueError(f"Row {row.name}: visit arrays mismatched.")

        for d, freq_val, hs_val in zip(dates, v, h):
            rows.append({"Patient": pid, "VisitDate": d, "Freq": freq_val, "HasSz": hs_val})

    pv = pd.DataFrame(rows)
    if pv.empty:
        return pv

    # Collapse to unique patient-date
    def agg_freq(x):
        v = x[np.isfinite(x)]
        return np.nanmean(v) if len(v) > 0 else np.nan

    def agg_hasz(x):
        v = x[np.isfinite(x)]
        return np.nanmax(v) if len(v) > 0 else np.nan

    grp = pv.groupby(["Patient", "VisitDate"])
    vuniq = grp["Freq"].agg(agg_freq).reset_index(name="Freq")
    vuniq["HasSz"] = grp["HasSz"].agg(agg_hasz).values

    # Rule 1: HasSz==0 with no documented frequency → impute SzFreq=0
    vuniq["Freq_R1"] = vuniq["Freq"].copy()
    mask_r1 = ~np.isfinite(vuniq["Freq_R1"].values) & (vuniq["HasSz"].values == 0)
    vuniq.loc[mask_r1, "Freq_R1"] = 0.0

    # Rule 2: documented SzFreq with no HasSz → impute HasSz
    
    freq_r1 = vuniq["Freq_R1"].to_numpy(dtype=float)
    has_sz = vuniq["HasSz"].to_numpy(dtype=float)

    mask_r2_pos = (
        np.isfinite(freq_r1)
        & (freq_r1 > 0)
        & ~np.isfinite(has_sz)
    )
    
    mask_r2_zer = np.isfinite(vuniq["Freq_R1"].values) & (vuniq["Freq_R1"].values == 0) & ~np.isfinite(vuniq["HasSz"].values)
    vuniq.loc[mask_r2_pos, "HasSz"] = 1.0
    vuniq.loc[mask_r2_zer, "HasSz"] = 0.0
    print(f"[Rule 2] Imputed HasSz for {mask_r2_pos.sum()} visits with SzFreq>0, "
          f"{mask_r2_zer.sum()} visits with SzFreq=0")
    return vuniq


# ─────────────────────────────────────────────
# PATIENT SEIZURE METRICS
# ─────────────────────────────────────────────

def build_patient_seizure_metrics(vuniq: pd.DataFrame) -> pd.DataFrame:
    def frac_hasz1(x):
        v = x[np.isfinite(x)]
        valid = v[(v == 0) | (v == 1)]
        if len(valid) == 0:
            return np.nan
        return np.sum(valid == 1) / len(valid)

    grp = vuniq.groupby("Patient")
    mean_sz  = grp["Freq_R1"].apply(lambda x: np.nanmean(x[np.isfinite(x)]) if np.isfinite(x).any() else np.nan)
    frac_hz  = grp["HasSz"].apply(frac_hasz1)
    szp = pd.DataFrame({"Patient": mean_sz.index, "MeanSzFreq": mean_sz.values, "FracVisits_HasSz1": frac_hz.values})
    return szp


# ─────────────────────────────────────────────
# PATIENT EPILEPSY TYPING
# ─────────────────────────────────────────────

def build_patient_typing(report_df: pd.DataFrame, canonical3: list) -> pd.DataFrame:
    pid   = report_df["patient_id"].values
    etype = report_df["epilepsy_type"].str.strip().fillna("")
    espec = report_df["epilepsy_specific"].str.strip().fillna("")

    ok_t = etype != ""
    ok_s = espec != ""

    def unique_val(vals):
        vals = [v for v in vals if v != ""]
        u = list(set(vals))
        if len(u) == 0:
            return ""
        if len(u) > 1:
            raise ValueError(f"Conflicting values: {u}")
        return u[0]

    # epilepsy_type per patient
    
    df_t = pd.DataFrame({
    "Patient": pid[ok_t.to_numpy()],
    "EpilepsyType_raw": etype[ok_t.to_numpy()]
    })
    
    etype_per = df_t.groupby("Patient")["EpilepsyType_raw"].apply(unique_val).reset_index(name="EpilepsyType")

    # epilepsy_specific per patient
    
    df_s = pd.DataFrame({
    "Patient": pid[ok_s.to_numpy()],
    "EpilepsySpecific_raw": espec[ok_s.to_numpy()]
    })
    
    espec_per = df_s.groupby("Patient")["EpilepsySpecific_raw"].apply(unique_val).reset_index(name="EpilepsySpecific")

    T = etype_per.merge(espec_per, on="Patient", how="outer")

    spec_norm = T["EpilepsySpecific"].str.lower().str.strip().fillna("")
    type_norm = T["EpilepsyType"].str.lower().str.strip().fillna("")

    epi_type3 = pd.Series([""] * len(T), index=T.index)
    epi_type3[spec_norm.str.contains("temporal")]                  = "Temporal"
    epi_type3[spec_norm.str.contains("frontal")]                   = "Frontal"
    epi_type3[(epi_type3 == "") & (type_norm == "general")]        = "General"

    T["EpiType3"] = pd.Categorical(epi_type3, categories=canonical3)
    return T


# ─────────────────────────────────────────────
# RESOLVE REPORTED SPIKE STATUS
# ─────────────────────────────────────────────

def resolve_reported_spike_status(report_df: pd.DataFrame) -> pd.DataFrame:
    any_spikes   = report_df["report_SPORADIC_EPILEPTIFORM_DISCHARGES"].fillna("").str.strip()
    is_main_pres = (any_spikes == "present")
    is_main_abs  = (any_spikes == "absent")

    def norm(col):
        return report_df[col].str.lower().str.strip().fillna("")

    raw_f = norm("jay_focal_epi")
    raw_m = norm("jay_multifocal_epi")
    raw_g = norm("jay_gen_epi")

    is_f_p = (raw_f == "present"); is_f_a = (raw_f == "absent")
    is_m_p = (raw_m == "present"); is_m_a = (raw_m == "absent")
    is_g_p = (raw_g == "present"); is_g_a = (raw_g == "absent")

    present_jay_any = is_f_p | is_m_p | is_g_p
    all_jay_absent  = is_f_a & is_m_a & is_g_a
    blank_main      = ~(is_main_pres | is_main_abs)
    blank_jay_all   = ~(is_f_p | is_f_a) & ~(is_m_p | is_m_a) & ~(is_g_p | is_g_a)

    rep = pd.Series([""] * len(report_df), index=report_df.index)
    rep[is_main_pres | present_jay_any] = "present"
    rep[all_jay_absent & blank_main]    = "absent"
    rep[is_main_abs   & blank_jay_all]  = "absent"
    rep[rep == ""]                      = "unknown"

    slim = report_df[["Patient", "Session"]].copy()
    slim["ReportStatus"] = pd.Categorical(rep, categories=["absent", "present", "unknown"])
    return slim


# ─────────────────────────────────────────────
# BUILD FILTERED VIEW
# ─────────────────────────────────────────────

def build_filtered_view(sessions_filtered, report_in, patient_typing_all,
                        sz_freq_per_patient, nesd_label, bad_types, canonical3, n_patients_total):
    # Join report to kept sessions
    sess_keys = sessions_filtered[["Patient", "Session"]].drop_duplicates()
    report_kept = report_in.merge(
        sess_keys.rename(columns={"Patient": "patient_id", "Session": "session_number"}),
        on=["patient_id", "session_number"]
    ).copy()
    report_kept["Patient"] = report_kept["patient_id"].astype(float)
    report_kept["Session"] = report_kept["session_number"].astype(float)

    patients_kept = sessions_filtered["Patient"].unique()
    n_after_outpt_routine = len(patients_kept)

    typing_filtered = patient_typing_all[patient_typing_all["Patient"].isin(patients_kept)].copy()

    # Patient-level mean spike rate
    mean_sr = sessions_filtered.groupby("Patient")["SpikeRate_perHour"].mean().reset_index(name="MeanSpikeRate_perHour")
    plsr = mean_sr.merge(typing_filtered[["Patient", "EpilepsyType", "EpilepsySpecific", "EpiType3"]], on="Patient")

    # Epilepsy masks
    etype_norm = plsr["EpilepsyType"].str.lower().str.strip().fillna("")
    is_nesd    = etype_norm == nesd_label.lower().strip()
    is_bad     = etype_norm.isin(bad_types) | (etype_norm == "")
    is_epilepsy = ~is_nesd & ~is_bad

    # Session-level spike rates
    slsr = sessions_filtered[["Patient", "Session", "SpikeRate_perHour"]].copy()
    slsr = slsr.rename(columns={"SpikeRate_perHour": "SpikesPerHour"})

    # Cohort: epilepsy + documented seizure frequency
    sz_freq_filtered = sz_freq_per_patient[sz_freq_per_patient["Patient"].isin(patients_kept)].copy()
    ep_patients = plsr.loc[is_epilepsy, ["Patient"]]
    sz_freq_epi = sz_freq_filtered.merge(ep_patients, on="Patient")

    n_patients_with_epilepsy = sz_freq_epi["Patient"].nunique()
    n_patients_with_sz_freq  = sz_freq_epi.loc[np.isfinite(sz_freq_epi["MeanSzFreq"]), "Patient"].nunique()

    cohort_patients = sz_freq_epi.loc[np.isfinite(sz_freq_epi["MeanSzFreq"]), "Patient"].unique()
    cohort_table    = pd.DataFrame({"Patient": cohort_patients})

    # Restrict all tables to cohort
    plsr          = plsr[plsr["Patient"].isin(cohort_patients)].copy().reset_index(drop=True)
    typing_filtered = typing_filtered[typing_filtered["Patient"].isin(cohort_patients)].copy().reset_index(drop=True)
    sessions_filt2  = sessions_filtered[sessions_filtered["Patient"].isin(cohort_patients)].copy().reset_index(drop=True)
    report_kept2    = report_kept[report_kept["Patient"].isin(cohort_patients)].copy().reset_index(drop=True)
    sz_freq_epi     = sz_freq_epi[sz_freq_epi["Patient"].isin(cohort_patients)].copy().reset_index(drop=True)

    etype_norm2 = plsr["EpilepsyType"].str.lower().str.strip().fillna("")
    is_epilepsy2 = ~(etype_norm2 == nesd_label.lower().strip()) & ~(etype_norm2.isin(bad_types) | (etype_norm2 == ""))

    print(f"[Cohort] {len(cohort_patients)} epilepsy patients with documented seizure frequency")

    # Canonical-subtype typing for Spearman
    keep_canon3 = (
        plsr["EpiType3"].notna() &
        plsr["EpiType3"].astype(str).isin(canonical3) &
        is_epilepsy2
    )
    typed_patients = plsr.loc[keep_canon3, "Patient"].values
    typing_canon   = typing_filtered[typing_filtered["Patient"].isin(typed_patients)].copy()

    # Spearman input tables
    pss_all = plsr.loc[is_epilepsy2, ["Patient", "MeanSpikeRate_perHour"]].merge(sz_freq_epi, on="Patient")
    keep_all = np.isfinite(pss_all["MeanSpikeRate_perHour"].values) & np.isfinite(pss_all["MeanSzFreq"].values)
    pss_all  = pss_all[keep_all].reset_index(drop=True)

    sz_freq_canon = sz_freq_epi.merge(
        plsr.loc[is_epilepsy2 & keep_canon3, ["Patient", "EpiType3"]], on="Patient"
    )
    pss_typed = plsr.loc[
        is_epilepsy2 & plsr["Patient"].isin(typed_patients),
        ["Patient", "MeanSpikeRate_perHour"]
    ].merge(sz_freq_canon, on="Patient")
    keep_typed = (
        np.isfinite(pss_typed["MeanSpikeRate_perHour"].values) &
        np.isfinite(pss_typed["MeanSzFreq"].values) &
        pss_typed["EpiType3"].notna() &
        pss_typed["EpiType3"].astype(str).isin(canonical3)
    )
    pss_typed = pss_typed[keep_typed].reset_index(drop=True)

    # Canonical3 group stats for Fig 1B
    c3_subset = plsr.loc[is_epilepsy2 & keep_canon3, ["Patient", "EpiType3", "MeanSpikeRate_perHour"]].copy()
    c3_subset = c3_subset.rename(columns={"EpiType3": "EpiType4"})

    c3_stats = c3_subset.groupby("EpiType4")["MeanSpikeRate_perHour"].agg(
        GroupCount=lambda x: np.isfinite(x).sum(),
        Median=lambda x: np.nanmedian(x[np.isfinite(x)]),
        P25=lambda x: np.nanpercentile(x[np.isfinite(x)], 25),
        P75=lambda x: np.nanpercentile(x[np.isfinite(x)], 75),
    ).reset_index()

    c3_pairs = [["General", "Temporal"], ["General", "Frontal"], ["Temporal", "Frontal"]]
    p_pair = []
    for g1, g2 in c3_pairs:
        xa = c3_subset.loc[c3_subset["EpiType4"].astype(str) == g1, "MeanSpikeRate_perHour"].dropna().values
        xb = c3_subset.loc[c3_subset["EpiType4"].astype(str) == g2, "MeanSpikeRate_perHour"].dropna().values
        if len(xa) >= 3 and len(xb) >= 3:
            _, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        else:
            p = np.nan
        p_pair.append(p)
    p_pair = np.array(p_pair)
    p_pair_bonf = np.minimum(p_pair * 3, 1.0)

    # Exclusion counts
    ec = {
        "nTotal":               n_patients_total,
        "nAfterOutptRoutine":   n_after_outpt_routine,
        "nExcludedNoEpilepsy":  n_after_outpt_routine - n_patients_with_epilepsy,
        "nExcludedNoSzFreq":    n_patients_with_epilepsy - n_patients_with_sz_freq,
        "nFinalCohort":         len(cohort_patients),
    }
    assert (ec["nExcludedNoEpilepsy"] + ec["nExcludedNoSzFreq"] + ec["nFinalCohort"]
            == ec["nAfterOutptRoutine"]), "Flow count mismatch"

    views = {
        "SessionsForFigures":        sessions_filt2,
        "ReportForKeptSessions":     report_kept2,
        "PatientTypingFiltered":     typing_canon,
        "PatientTyping_AllEpilepsy": typing_filtered,
        "SessionLevelSpikeRates":    slsr,
        "PatientLevelSpikeRates":    plsr,
        "PatientSpikeSz_All":        pss_all,
        "PatientSpikeSz_Typed":      pss_typed,
        "IsEpilepsyMask":            is_epilepsy2.values,
        "Canonical3_SubsetTable":    c3_subset,
        "Canonical3_Stats":          c3_stats,
        "Canonical3_Pairs":          c3_pairs,
        "PvalsPairwise":             p_pair,
        "PvalsPairwiseBonf":         p_pair_bonf,
        "ExclusionCounts":           ec,
    }
    return views


# ─────────────────────────────────────────────
# BUILD EEG-VISIT PAIR TABLE
# ─────────────────────────────────────────────

def build_eeg_visit_pairs(vuniq, session_level_spike_rates, report_kept, patient_typing):
    EPS_SPIKE = 1e-3

    # EEG dates
    eeg_raw = report_kept["start_time_deid"].str.strip()
    eeg_dt = pd.to_datetime(eeg_raw, errors="coerce")
    eeg_tbl = pd.DataFrame({
        "Patient": report_kept["Patient"].values,
        "Session": report_kept["Session"].values,
        "EEG_Date": eeg_dt.values,
    })
    eeg_tbl = eeg_tbl[~pd.isna(eeg_tbl["EEG_Date"])]
    eeg_tbl = eeg_tbl.merge(
        session_level_spike_rates[["Patient", "Session", "SpikesPerHour"]],
        on=["Patient", "Session"]
    )

    # Typed visits
    v_typed = vuniq[["Patient", "VisitDate", "Freq_R1", "HasSz"]].merge(
        patient_typing[["Patient", "EpiType3"]], on="Patient"
    )

    patients = np.intersect1d(eeg_tbl["Patient"].unique(), v_typed["Patient"].unique())

    records = []
    for p in patients:
        e_rows = eeg_tbl[eeg_tbl["Patient"] == p]
        v_rows = v_typed[v_typed["Patient"] == p]
        for _, erow in e_rows.iterrows():
            for _, vrow in v_rows.iterrows():
                lag = (vrow["VisitDate"] - erow["EEG_Date"]).days
                records.append({
                    "Patient": p,
                    "Session": erow["Session"],
                    "VisitDate": vrow["VisitDate"],
                    "SpikesPerHour": erow["SpikesPerHour"],
                    "SzFreq": vrow["Freq_R1"],
                    "HasSz": vrow["HasSz"],
                    "SignedLag_days": lag,
                    "EpiType3": str(vrow["EpiType3"]),
                })

    pt = pd.DataFrame(records)
    if pt.empty:
        raise ValueError("No EEG-visit pairs built.")

    pt["EEG_ID"] = pt["Patient"].astype(str) + "_" + pt["Session"].astype(str)

    keep = (
        np.isfinite(pt["SpikesPerHour"].values) &
        np.isfinite(pt["SzFreq"].values) &
        np.isfinite(pt["SignedLag_days"].values) &
        (pt["EpiType3"].str.len() > 0)
    )
    n_before = len(pt)
    pt = pt[keep].reset_index(drop=True)
    print(f"[build_eeg_visit_pairs] {len(patients)} patients, {len(pt)} pairs "
          f"({n_before - len(pt)} removed)")

    pt["LogSpikesPerHour"] = np.log(pt["SpikesPerHour"].values + EPS_SPIKE)
    pt["SignedLag_years"]  = pt["SignedLag_days"].values / 365.25
    pt["PatientID"]        = pt["Patient"].astype(str)
    return pt


# ─────────────────────────────────────────────
# MIXED EFFECTS MODELS (statsmodels GLMM via GEE fallback)
# ─────────────────────────────────────────────
# NOTE: Python's statsmodels does not have a direct GLMM equivalent to
# MATLAB's fitglme with Laplace approximation. We use GEE (Generalized
# Estimating Equations) as the closest available alternative, which gives
# population-average estimates with robust SEs accounting for clustering.
# For the same Laplace GLMM, consider the `lme4` R package via rpy2.

def fit_mixed_effects_models(pair_table, n_boot=N_BOOT, alpha=ALPHA):
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.cov_struct import Exchangeable

    canonical3 = ["Frontal", "General", "Temporal"]

    keep = (
        pair_table["EpiType3"].isin(canonical3) &
        np.isfinite(pair_table["LogSpikesPerHour"].values) &
        np.isfinite(pair_table["SignedLag_years"].values) &
        np.isfinite(pair_table["HasSz"].values)
    )
    T = pair_table[keep].copy().reset_index(drop=True)

    T["HasSz_bin"]     = (T["HasSz"] == 1).astype(float)
    T["AbsLag_years"]  = T["SignedLag_years"].abs()
    T["VisitAfterEEG"] = (T["SignedLag_years"] >= 0).astype(float)

    # reference: Temporal
    T["EpiType3_Frontal"] = (T["EpiType3"] == "Frontal").astype(float)
    T["EpiType3_General"] = (T["EpiType3"] == "General").astype(float)

    # Interaction terms
    T["SpikexLag"]  = T["LogSpikesPerHour"] * T["AbsLag_years"]
    T["SpikexDir"]  = T["LogSpikesPerHour"] * T["VisitAfterEEG"]

    print(f"[Model table] {len(T)} pairs, {T['Patient'].nunique()} patients")

    def fit_gee(df, formula_cols, group_col="PatientID"):
        # Build design matrix manually
        y = df["HasSz_bin"].values
        X = np.column_stack([np.ones(len(df))] + [df[c].values for c in formula_cols])
        groups = df[group_col].values
        try:
            model = GEE(y, X, groups, family=Binomial(), cov_struct=Exchangeable())
            result = model.fit(maxiter=60, ctol=1e-6)
            return result, formula_cols
        except Exception as e:
            print(f"  GEE failed: {e}")
            return None, formula_cols

    # M1: with interactions
    cols_M1 = ["LogSpikesPerHour", "AbsLag_years", "VisitAfterEEG",
               "EpiType3_Frontal", "EpiType3_General", "SpikexLag", "SpikexDir"]
    mdl_M1, _ = fit_gee(T, cols_M1)

    # M2: without interactions
    cols_M2 = ["LogSpikesPerHour", "AbsLag_years", "VisitAfterEEG",
               "EpiType3_Frontal", "EpiType3_General"]
    mdl_M2, _ = fit_gee(T, cols_M2)

    def extract_fe(mdl, col_names):
        if mdl is None:
            return None
        names = ["(Intercept)"] + list(col_names)
        betas = mdl.params
        ses   = mdl.bse
        pvals = mdl.pvalues
        return pd.DataFrame({
            "Term":   names[:len(betas)],
            "Beta":   betas,
            "SE":     ses,
            "p":      pvals,
            "OR":     np.exp(betas),
            "OR_lo":  np.exp(betas - 1.96 * ses),
            "OR_hi":  np.exp(betas + 1.96 * ses),
        })

    fe_M1 = extract_fe(mdl_M1, cols_M1)
    fe_M2 = extract_fe(mdl_M2, cols_M2)

    if fe_M1 is not None:
        print("\nM1 fixed effects:"); print(fe_M1)
    if fe_M2 is not None:
        print("\nM2 fixed effects:"); print(fe_M2)

    # Directional models
    T_after  = T[T["SignedLag_years"] >= 0].copy()
    T_before = T[T["SignedLag_years"] <= 0].copy()
    T_after["SpikexSignedLag"]  = T_after["LogSpikesPerHour"]  * T_after["SignedLag_years"]
    T_before["SpikexSignedLag"] = T_before["LogSpikesPerHour"] * T_before["SignedLag_years"]
    cols_dir = ["LogSpikesPerHour", "SignedLag_years", "EpiType3_Frontal", "EpiType3_General", "SpikexSignedLag"]

    mdl_after,  _ = fit_gee(T_after,  cols_dir)
    mdl_before, _ = fit_gee(T_before, cols_dir)
    fe_after  = extract_fe(mdl_after,  cols_dir)
    fe_before = extract_fe(mdl_before, cols_dir)

    # Bootstrap (patient-level)
    def run_bootstrap(df, col_names, n_b, a, label):
        if mdl_M1 is None or n_b == 0:
            return None, None, 0, 0
        patients = df["PatientID"].unique()
        n_pat    = len(patients)
        n_fixed  = len(col_names) + 1
        betas    = np.full((n_b, n_fixed), np.nan)
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.cov_struct import Exchangeable
        print(f"Bootstrapping {label} ({n_b} iterations)...")
        counter = 0
        for b in range(n_b):
            idx = np.random.randint(0, n_pat, n_pat)
            boot_pats = patients[idx]
            parts = []
            for ki, bp in enumerate(boot_pats):
                sub = df[df["PatientID"] == bp].copy()
                sub["PatientID"] = str(ki)
                parts.append(sub)
            Tb = pd.concat(parts, ignore_index=True)
            y_b = Tb["HasSz_bin"].values
            X_b = np.column_stack([np.ones(len(Tb))] + [Tb[c].values for c in col_names])
            try:
                m = GEE(y_b, X_b, Tb["PatientID"].values, family=Binomial(), cov_struct=Exchangeable())
                r = m.fit(maxiter=60, ctol=1e-6)
                betas[b] = r.params
            except Exception:
                pass
            counter += 1
            print(counter)

        converged = np.all(np.isfinite(betas), axis=1)
        betas_c = betas[converged]
        print(f"{label} bootstrap: {converged.sum()}/{n_b} converged "
              f"({100*converged.mean():.1f}%)")

        if converged.sum() == 0:
            return None, None, 0, n_b

        fe_all = extract_fe(mdl_M1 if label.startswith("M1") else mdl_M2 if label.startswith("M2") else None, col_names)
        ci_lo = np.percentile(betas_c, 100 * a / 2,     axis=0)
        ci_hi = np.percentile(betas_c, 100 * (1 - a/2), axis=0)

        if fe_all is not None:
            beta_obs = fe_all["Beta"].values
        else:
            beta_obs = np.nanmean(betas_c, axis=0)

        boot_p = np.array([
            min(2 * min(np.mean(betas_c[:, k] <= 0), np.mean(betas_c[:, k] >= 0)), 1.0)
            for k in range(betas_c.shape[1])
        ])
        names = ["(Intercept)"] + list(col_names)
        t_boot = pd.DataFrame({
            "Term":      names[:len(beta_obs)],
            "Beta":      beta_obs,
            "Boot_CI_lo": ci_lo,
            "Boot_CI_hi": ci_hi,
            "OR":        np.exp(beta_obs),
            "OR_CI_lo":  np.exp(ci_lo),
            "OR_CI_hi":  np.exp(ci_hi),
            "Boot_p":    boot_p,
        })
        print(f"{label} bootstrapped ORs:"); print(t_boot)
        return t_boot, betas_c, converged.sum(), n_b

    t_boot1, bb1, nc1, nt1 = run_bootstrap(T, cols_M1, n_boot, alpha, "M1")
    t_boot2, bb2, nc2, nt2 = run_bootstrap(T, cols_M2, n_boot, alpha, "M2")

    mmr = {
        "ModelTable":   T,
        "mdl_M1":       mdl_M1,
        "mdl_M2":       mdl_M2,
        "FE_M1":        fe_M1,
        "FE_M2":        fe_M2,
        "BootstrapTable1": t_boot1,
        "BootstrapBetas1": bb1,
        "BootstrapTable2": t_boot2,
        "BootstrapBetas2": bb2,
        "LRT_p":        np.nan,   # LRT not straightforward with GEE
        "BootstrapConvergence": {
            "M1_nConverged": nc1, "M1_nTotal": nt1,
            "M2_nConverged": nc2, "M2_nTotal": nt2,
        },
        "mdl_after":    mdl_after,
        "mdl_before":   mdl_before,
        "FE_after":     fe_after,
        "FE_before":    fe_before,
        "BootstrapTable_after":  None,
        "BootstrapTable_before": None,
        "cols_M1": cols_M1,
    }
    print("\nDone. Primary model: M1 (GEE logistic + subtypes + interactions).")
    return mmr


# ─────────────────────────────────────────────
# FIGURE 1
# ─────────────────────────────────────────────

def make_fig1_controls(views, eps_rate=EPS_RATE, y_zero=Y_ZERO, y_lims=Y_LIMS,
                       title_y_offset=TITLE_Y_OFFSET, n_boot=N_BOOT, alpha=ALPHA):
    session_level = views["SessionLevelSpikeRates"]
    report        = views["ReportForKeptSessions"]

    report_slim = resolve_reported_spike_status(report)
    join_a = session_level[["Patient", "Session", "SpikesPerHour"]].merge(
        report_slim, on=["Patient", "Session"]
    )

    x_abs = join_a.loc[join_a["ReportStatus"] == "absent",  "SpikesPerHour"].dropna().values
    x_pre = join_a.loc[join_a["ReportStatus"] == "present", "SpikesPerHour"].dropna().values

    _, p_a = stats.mannwhitneyu(x_abs, x_pre, alternative="two-sided")
    effect_a = cliff_delta(x_pre, x_abs)

    med_abs, lo_abs, hi_abs = bootstrap_median_ci(x_abs, n_boot, alpha)
    med_pre, lo_pre, hi_pre = bootstrap_median_ci(x_pre, n_boot, alpha)

    all_sr  = np.concatenate([x_abs, x_pre])
    all_g   = np.concatenate([["Absent"] * len(x_abs), ["Present"] * len(x_pre)])
    Y_A_raw = to_log10_per_hour(all_sr, eps_rate)
    Y_A     = add_y_jitter_eps(Y_A_raw, np.log10(eps_rate), y_lims, 0.02)

    sub = views["PatientSpikeSz_Typed"][["EpiType3", "MeanSpikeRate_perHour"]].copy()
    sub = sub.rename(columns={"EpiType3": "EpiType4"})
    Y_C_raw = to_log10_per_hour(sub["MeanSpikeRate_perHour"].values, eps_rate)
    Y_C     = add_y_jitter_eps(Y_C_raw, y_zero, y_lims, 0.02)

    groups = sub["EpiType4"].astype(str).values
    cats   = [c for c in CANONICAL3 if c in np.unique(groups)]
    kw_data = [sub.loc[groups == c, "MeanSpikeRate_perHour"].dropna().values for c in cats]
    h_stat, p_kw = kruskal(*kw_data)
    # eta-squared from KW
    n_tot = sum(len(g) for g in kw_data)
    eta2_kw = (h_stat - len(kw_data) + 1) / (n_tot - len(kw_data))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("white")

    # Panel A
    ax = axes[0]
    ax.set_facecolor("white")
    groups_cat = pd.Categorical(all_g, categories=["Absent", "Present"])
    for xi, (grp, color) in enumerate(zip(["Absent", "Present"], ["#4878CF", "#D65F5F"])):
        mask = np.array(all_g) == grp
        y_g = Y_A[mask]
        bp = ax.boxplot(y_g, positions=[xi + 1], widths=0.4, patch_artist=True,
                        showfliers=False, medianprops=dict(color="k"))
        for patch in bp["boxes"]:
            patch.set_facecolor(color); patch.set_alpha(0.3)
        jitter = np.random.uniform(-0.15, 0.15, len(y_g))
        ax.scatter(xi + 1 + jitter, y_g, alpha=0.2, s=18, color=color)

    ax.axhline(y_zero, linestyle=":", color="gray", linewidth=1.2)
    ax.set_ylim(y_lims)
    set_log10_ticks(ax, "y", eps_rate, y_lims)
    ax.set_ylabel("Spikes/hour (log scale)", fontsize=14)
    add_median_ci_overlay(ax, 1, med_abs, lo_abs, hi_abs, eps_rate)
    add_median_ci_overlay(ax, 2, med_pre, lo_pre, hi_pre, eps_rate)
    add_sigbar(ax, 1, 2, y_lims[1] - 0.08 * (y_lims[1] - y_lims[0]), p_label(p_a))
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Absent (N={len(x_abs)})", f"Present (N={len(x_pre)})"], rotation=20, fontsize=11)
    ax.set_title("A. Reported presence or absence of spikes", fontsize=14)
    ax.grid(True, alpha=0.3); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel B
    ax2 = axes[1]
    ax2.set_facecolor("white")
    c3_stats = views["Canonical3_Stats"]
    colors_c3 = {"General": "#0072BD", "Temporal": "#D95319", "Frontal": "#EDB120"}
    for xi, cat in enumerate(cats):
        mask = groups == cat
        y_g  = Y_C[mask]
        col  = colors_c3.get(cat, "gray")
        bp = ax2.boxplot(y_g, positions=[xi + 1], widths=0.4, patch_artist=True,
                         showfliers=False, medianprops=dict(color="k"))
        for patch in bp["boxes"]:
            patch.set_facecolor(col); patch.set_alpha(0.3)
        jitter = np.random.uniform(-0.15, 0.15, len(y_g))
        ax2.scatter(xi + 1 + jitter, y_g, alpha=0.2, s=18, color=col)
        xg = sub.loc[groups == cat, "MeanSpikeRate_perHour"].values
        mg, log, hig = bootstrap_median_ci(xg, n_boot, alpha)
        add_median_ci_overlay(ax2, xi + 1, mg, log, hig, eps_rate)

    ax2.axhline(y_zero, linestyle=":", color="gray", linewidth=1.2)
    ax2.set_ylim(y_lims)
    set_log10_ticks(ax2, "y", eps_rate, y_lims)
    ax2.set_ylabel("Spikes/hour (log scale)", fontsize=14)

    p_pair_bonf = views["PvalsPairwiseBonf"]
    c3_pairs    = views["Canonical3_Pairs"]
    y0 = y_lims[1]; ystep = 0.08 * (y_lims[1] - y_lims[0])
    y0_start = y_lims[1] - 0.05 * (y_lims[1] - y_lims[0])
    for ii, (g1, g2) in enumerate(c3_pairs):
        if g1 in cats and g2 in cats:
            x1i = cats.index(g1) + 1
            x2i = cats.index(g2) + 1
            p   = p_pair_bonf[ii]
            if np.isnan(p):
                continue
            lbl = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
            add_sigbar(ax2, x1i, x2i, y0_start - ii * ystep, lbl)

    n_per_cat = {c: int(c3_stats.loc[c3_stats["EpiType4"].astype(str) == c, "GroupCount"].values[0])
                 if c in c3_stats["EpiType4"].astype(str).values else 0 for c in cats}
    ax2.set_xticks(range(1, len(cats) + 1))
    ax2.set_xticklabels([f"{c} (N={n_per_cat.get(c,0)})" for c in cats], rotation=20, fontsize=11)
    ax2.set_title("B. Epilepsy subtype", fontsize=14)
    ax2.grid(True, alpha=0.3); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.tight_layout()

    fig1_stats = {
        "p_rankSum_A":   p_a,
        "effectA_cliff": effect_a,
        "m_pre": med_pre, "lo_pre": lo_pre, "hi_pre": hi_pre,
        "m_abs": med_abs, "lo_abs": lo_abs, "hi_abs": hi_abs,
        "p_kw_C":    p_kw,
        "eta2_kw_C": eta2_kw,
        "p_pair_bonf": p_pair_bonf,
    }
    return fig, fig1_stats


# ─────────────────────────────────────────────
# FIGURE 2 / SPEARMAN
# ─────────────────────────────────────────────

def spearman_plotting_function(pss_all, pss_typed, canonical3, x_lims, y_lims,
                               fig_out, freq_field="MeanSzFreq", label_suffix="", non_zero_only=False):
    n_boot = 5000; alpha = 0.05; font_l = 14
    COL_all   = (0.45, 0.45, 0.45)
    COL_front = (0.93, 0.69, 0.13)
    COL_temp  = (0.85, 0.33, 0.10)
    COL_gen   = (0.00, 0.45, 0.74)

    x_all = pss_all["MeanSpikeRate_perHour"].values.astype(float)
    y_all = pss_all[freq_field].values.astype(float)
    mask_all = np.isfinite(x_all) & np.isfinite(y_all)
    if non_zero_only:
        mask_all &= (x_all > 0) & (y_all > 0)
    n_all = mask_all.sum()

    if n_all >= 3:
        rs_all, p_all = stats.spearmanr(x_all[mask_all], y_all[mask_all])
        rho_hat, rho_lo, rho_hi = bootstrap_spearman_ci(x_all[mask_all], y_all[mask_all], n_boot, alpha)
    else:
        rs_all = p_all = rho_lo = rho_hi = np.nan

    rows_out = []
    subtype_ci = pd.DataFrame({
        "Group": canonical3,
        "rho": [np.nan]*3, "ci_lo": [np.nan]*3, "ci_hi": [np.nan]*3
    })

    for ii, g in enumerate(canonical3):
        m_base = pss_typed["EpiType3"].astype(str) == g
        xg = pss_typed.loc[m_base, "MeanSpikeRate_perHour"].values.astype(float)
        yg = pss_typed.loc[m_base, freq_field].values.astype(float)
        mask = np.isfinite(xg) & np.isfinite(yg)
        if non_zero_only:
            mask &= (xg > 0) & (yg > 0)
        n = mask.sum()
        if n >= 3:
            rs, p = stats.spearmanr(xg[mask], yg[mask])
            rho_g, lo, hi = bootstrap_spearman_ci(xg[mask], yg[mask], n_boot, alpha)
        else:
            rs = p = rho_g = lo = hi = np.nan
        subtype_ci.loc[ii, ["rho", "ci_lo", "ci_hi"]] = [rho_g, lo, hi]
        rows_out.append({"Group": g, "N": n, "Spearman_r": rs, "p_raw": p})

    sr = pd.DataFrame(rows_out)
    sr["p_bonf"] = (sr["p_raw"] * len(sr)).clip(upper=1.0)
    
    # epsilons
    x_used = x_all[mask_all]; y_used = y_all[mask_all]
    pos_r = x_used[x_used > 0]; pos_s = y_used[y_used > 0]
    eps_rate_sp = 0.5 * (pos_r.min() if len(pos_r) else 1e-6)
    eps_sz_sp   = 0.5 * (pos_s.min() if len(pos_s) else 1e-6)
    x_zero = np.log10(eps_sz_sp); y_zero_sp = np.log10(eps_rate_sp)

    logX_all = np.log10(x_used + (x_used <= 0) * eps_rate_sp)
    logY_all = np.log10(y_used + (y_used <= 0) * eps_sz_sp)
    is_zx_all = (x_used == 0); is_zy_all = (y_used == 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    ax_a = axes[0, 0]

    ax_a.axvline(x_zero,   linestyle=":", color="gray", linewidth=1.2)
    ax_a.axhline(y_zero_sp, linestyle=":", color="gray", linewidth=1.2)
    nz = ~is_zx_all & ~is_zy_all
    ax_a.scatter(logX_all[nz], logY_all[nz], alpha=0.25, s=14, color=COL_all)
    ax_a.scatter(logX_all[~nz], logY_all[~nz], marker="*", s=40, color=COL_all, alpha=0.5)

    if n_all >= 3:
        Xfit = np.column_stack([np.ones(n_all), logX_all])
        b    = np.linalg.lstsq(Xfit, logY_all, rcond=None)[0]
        xgrid = np.linspace(x_lims[0], x_lims[1], 300)
        ax_a.plot(xgrid, b[0] + b[1] * xgrid, "k-", linewidth=2)

    ax_a.set_xlim(x_lims); ax_a.set_ylim(y_lims)
    ax_a.set_xlabel("Seizures per month (log scale)", fontsize=font_l)
    ax_a.set_ylabel("Spikes per hour (log scale)", fontsize=font_l)
    set_log10_ticks(ax_a, "x", eps_sz_sp, x_lims)
    set_log10_ticks(ax_a, "y", eps_rate_sp, y_lims)
    ax_a.set_title(f"A. All epilepsy{label_suffix} (N={n_all})", fontsize=font_l, fontweight="bold")
    ax_a.text(0.98, 0.95,
              f"ρ={rs_all:.2f} [{rho_lo:.2f}-{rho_hi:.2f}], {p_label(p_all)}",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=font_l - 2, fontweight="bold")
    ax_a.grid(True, alpha=0.3)

    col_map = {"Frontal": COL_front, "Temporal": COL_temp, "General": COL_gen}
    panel_order  = ["Frontal", "Temporal", "General"]
    panel_titles = ["B. Frontal", "C. Temporal", "D. General"]
    ax_list = [axes[0, 1], axes[1, 0], axes[1, 1]]

    for pi, (g, ptitle) in enumerate(zip(panel_order, panel_titles)):
        ax = ax_list[pi]
        col = col_map[g]
        idx = (pss_typed["EpiType3"].astype(str) == g) & \
              np.isfinite(pss_typed["MeanSpikeRate_perHour"].values) & \
              np.isfinite(pss_typed[freq_field].values)
        if non_zero_only:
            idx &= (pss_typed["MeanSpikeRate_perHour"].values > 0) & (pss_typed[freq_field].values > 0)

        if idx.sum() == 0:
            ax.axis("off"); continue

        xr = pss_typed.loc[idx, freq_field].values.astype(float)
        yr = pss_typed.loc[idx, "MeanSpikeRate_perHour"].values.astype(float)
        logXg = np.log10(xr + (xr <= 0) * eps_sz_sp)
        logYg = np.log10(yr + (yr <= 0) * eps_rate_sp)
        is_zxg = (xr == 0); is_zyg = (yr == 0)

        ax.axvline(x_zero,    linestyle=":", color="gray", linewidth=1.2)
        ax.axhline(y_zero_sp, linestyle=":", color="gray", linewidth=1.2)
        nz = ~is_zxg & ~is_zyg
        ax.scatter(logXg[nz], logYg[nz], alpha=0.35, s=18, color=col)
        ax.scatter(logXg[~nz], logYg[~nz], marker="*", s=50, color=col, alpha=0.6)

        if nz.sum() >= 3:
            Xfit = np.column_stack([np.ones(nz.sum()), logXg[nz]])
            bg   = np.linalg.lstsq(Xfit, logYg[nz], rcond=None)[0]
            xgrid = np.linspace(x_lims[0], x_lims[1], 250)
            ax.plot(xgrid, bg[0] + bg[1] * xgrid, "-", color=col, linewidth=2)

        ax.set_xlim(x_lims); ax.set_ylim(y_lims)
        ax.set_xlabel("Seizures per month (log scale)", fontsize=font_l)
        ax.set_ylabel("Spikes per hour (log scale)", fontsize=font_l)
        set_log10_ticks(ax, "x", eps_sz_sp, x_lims)
        set_log10_ticks(ax, "y", eps_rate_sp, y_lims)

        row   = sr[sr["Group"] == g].iloc[0]
        rowci = subtype_ci[subtype_ci["Group"] == g].iloc[0]
        txt   = f"ρ={row.Spearman_r:.2f} [{rowci.ci_lo:.2f}-{rowci.ci_hi:.2f}], p_bonf{p_label(row.p_bonf).replace('p','')}"
        ax.set_title(f"{ptitle}{label_suffix} (N={int(row.N)})", fontsize=font_l, fontweight="bold")
        ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=font_l - 3, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, fig_out)
    return sr, rs_all, p_all, n_all, rho_lo, rho_hi, subtype_ci


# ─────────────────────────────────────────────
# TABLE 1
# ─────────────────────────────────────────────

def build_table1_flat(views, sz_freq_per_patient, vuniq, eps_rate=EPS_RATE, n_boot=N_BOOT, alpha=ALPHA):
    rk = views["ReportForKeptSessions"]
    pl = views["PatientLevelSpikeRates"]
    all_patients = pl["Patient"].values
    N_total = len(all_patients)

    # Age
    birth_str = rk["deid_birth_date"].str.strip().fillna("")
    bad_birth  = birth_str.isin(["", "null", "[null]"])
    ref_date   = pd.Timestamp("2000-01-01")
    birth_dt   = pd.to_datetime(birth_str.where(~bad_birth), format="%Y-%m-%d", errors="coerce")
    age_first_series = (ref_date - birth_dt).dt.days / 365.25
    rk2 = rk.copy(); rk2["_age"] = age_first_series.values
    age_per_pat = rk2.groupby("Patient")["_age"].min()
    age_vec = age_per_pat.reindex(all_patients).values
    age_vec = age_vec[np.isfinite(age_vec)]
    age_med = np.nanmedian(age_vec); age_q = np.nanpercentile(age_vec, [25, 75])

    # Sex
    sex_raw = rk["nlp_gender"].str.upper().str.strip().fillna("")
    sex_per = rk.assign(_sex=sex_raw).groupby("Patient")["_sex"].apply(
        lambda x: next((v for v in x if v in ("F", "M")), "")
    )
    sex_df = sex_per.reindex(all_patients).fillna("")
    n_f = (sex_df == "F").sum(); n_m = (sex_df == "M").sum(); n_u = N_total - n_f - n_m

    # Epilepsy subtype
    e3     = pl["EpiType3"].astype(str).values
    espec  = pl["EpilepsySpecific"].str.strip().fillna("").values
    is_ep  = views["IsEpilepsyMask"]
    is_temp  = (e3 == "Temporal")  & is_ep
    is_front = (e3 == "Frontal")   & is_ep
    is_gen   = (e3 == "General")   & is_ep
    is_canon = is_temp | is_front | is_gen
    is_unk   = is_ep & ~is_canon & np.isin(espec, ["", "Unclassified or Unspecified", "Unknown or MRN not found"])
    is_oth   = is_ep & ~is_canon & ~is_unk
    n_epi   = is_ep.sum()
    n_temp  = is_temp.sum(); n_front = is_front.sum(); n_gen = is_gen.sum()
    n_other = is_oth.sum();  n_subunk = is_unk.sum()

    # Visits
    n_visits = vuniq.groupby("Patient")["VisitDate"].nunique().reindex(all_patients).values
    vis_med = np.nanmedian(n_visits); vis_q = np.nanpercentile(n_visits, [25, 75])

    # Follow-up
    v_coh = vuniq[vuniq["Patient"].isin(all_patients)]
    fu = v_coh.groupby("Patient")["VisitDate"].agg(lambda x: (x.max() - x.min()).days / 365.25)
    fu_vec = fu.reindex(all_patients).values
    fu_med = np.nanmedian(fu_vec); fu_q = np.nanpercentile(fu_vec, [25, 75])

    # Doc rate
    doc_r = v_coh.groupby("Patient")["Freq_R1"].apply(lambda x: np.isfinite(x).mean())
    doc_vec = doc_r.reindex(all_patients).values * 100
    doc_med = np.nanmedian(doc_vec); doc_q = np.nanpercentile(doc_vec, [25, 75])

    # EEGs
    sess = views["SessionsForFigures"]
    n_eegs = sess.groupby("Patient")["Session"].nunique().reindex(all_patients).values
    eeg_med = np.nanmedian(n_eegs); eeg_q = np.nanpercentile(n_eegs, [25, 75])

    # Seizure freq
    szj = pd.DataFrame({"Patient": all_patients}).merge(sz_freq_per_patient, on="Patient")
    sf_vec = szj["MeanSzFreq"].dropna().values
    sf_vec = np.asarray(sf_vec, dtype=np.float64)

    sf_med = float(np.nanmedian(sf_vec))
    sf_q = np.nanpercentile(sf_vec, [25, 75]).astype(float)

    _, sf_lo, sf_hi = bootstrap_median_ci(sf_vec, n_boot, alpha)

    # Spike rate
    sr_vec = pl["MeanSpikeRate_perHour"].dropna().values
    sr_med = np.nanmedian(sr_vec); sr_q = np.nanpercentile(sr_vec, [25, 75])
    _, sr_lo, sr_hi = bootstrap_median_ci(sr_vec, n_boot, alpha)

    # EEG spike status
    rs_all = resolve_reported_spike_status(views["ReportForKeptSessions"])
    n_rep_pre = (rs_all["ReportStatus"] == "present").sum()
    n_rep_abs = (rs_all["ReportStatus"] == "absent").sum()
    n_rep_unk = (rs_all["ReportStatus"] == "unknown").sum()
    n_eegs_all = len(rs_all)

    # Patient-level spike status
    gp_rs = rs_all.groupby("Patient")["ReportStatus"]
    has_present = gp_rs.apply(lambda x: any(x == "present"))
    has_absent  = gp_rs.apply(lambda x: any(x == "absent"))
    pat_status  = pd.Series("unknown", index=has_present.index)
    pat_status[has_absent & ~has_present] = "absent"
    pat_status[has_present]               = "present"
    n_pats_pre = (pat_status == "present").sum()
    n_pats_abs = (pat_status == "absent").sum()
    n_pats_unk = (pat_status == "unknown").sum()
    n_pats_rs  = len(pat_status)

    def row(var, stat):
        return {"Variable": var, "Statistic": stat}

    rows = [
        row("Total N patients", str(N_total)),
        row("Age at first visit (years)", f"{age_med:.1f} ({age_q[0]:.1f}-{age_q[1]:.1f})"),
        row("Sex", ""),
        row("    Women",         f"{n_f} ({100*n_f/N_total:.1f}%)"),
        row("    Men",           f"{n_m} ({100*n_m/N_total:.1f}%)"),
        row("    Unknown/Other", f"{n_u} ({100*n_u/N_total:.1f}%)"),
        row("Epilepsy subtype", ""),
        row("    Temporal lobe", f"{n_temp}  ({100*n_temp /max(1,n_epi):.1f}%)"),
        row("    Frontal lobe",  f"{n_front} ({100*n_front/max(1,n_epi):.1f}%)"),
        row("    Generalized",   f"{n_gen}   ({100*n_gen  /max(1,n_epi):.1f}%)"),
        row("    Other",         f"{n_other} ({100*n_other/max(1,n_epi):.1f}%)"),
        row("    Unknown",       f"{n_subunk} ({100*n_subunk/max(1,n_epi):.1f}%)"),
        row("Number of clinic visits",               f"{vis_med:.1f} ({vis_q[0]:.1f}-{vis_q[1]:.1f})"),
        row("Follow-up duration (years)",            f"{fu_med:.1f} ({fu_q[0]:.1f}-{fu_q[1]:.1f})"),
        row("Visits with documented seizure frequency", f"{doc_med:.1f}% ({doc_q[0]:.1f}-{doc_q[1]:.1f})"),
        row("Number of EEGs",                        f"{eeg_med:.1f} ({eeg_q[0]:.1f}-{eeg_q[1]:.1f})"),
        row("Mean seizure frequency (seizures/month)", f"{sf_med:.2f} ({sf_q[0]:.2f}-{sf_q[1]:.2f}); median CI [{sf_lo:.2f}-{sf_hi:.2f}]"),
        row("Mean spike rate (spikes/hour)",          f"{sr_med:.2f} ({sr_q[0]:.2f}-{sr_q[1]:.2f}); median CI [{sr_lo:.2f}-{sr_hi:.2f}]"),
        row("EEGs with reported spikes", "N (% EEGs)"),
        row("    Present", f"{n_rep_pre} ({100*n_rep_pre/max(1,n_eegs_all):.1f}%)"),
        row("    Absent",  f"{n_rep_abs} ({100*n_rep_abs/max(1,n_eegs_all):.1f}%)"),
        row("    Unknown", f"{n_rep_unk} ({100*n_rep_unk/max(1,n_eegs_all):.1f}%)"),
        row("Patients with reported spikes", "N (% patients)"),
        row("    Present", f"{n_pats_pre} ({100*n_pats_pre/max(1,n_pats_rs):.1f}%)"),
        row("    Absent",  f"{n_pats_abs} ({100*n_pats_abs/max(1,n_pats_rs):.1f}%)"),
        row("    Unknown", f"{n_pats_unk} ({100*n_pats_unk/max(1,n_pats_rs):.1f}%)"),
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# COMPUTE VISIT-EEG GAPS
# ─────────────────────────────────────────────

def compute_visit_eeg_gaps(vuniq, report_kept):
    eeg_raw = report_kept["start_time_deid"].str.strip()
    eeg_dt  = pd.to_datetime(eeg_raw, format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    pid     = report_kept["Patient"].values
    eeg_tbl = pd.DataFrame({"Patient": pid, "EEG_Date": eeg_dt.values})
    eeg_tbl = eeg_tbl[~pd.isna(eeg_tbl["EEG_Date"])]

    gap_tbl = vuniq[["Patient", "VisitDate"]].copy()
    gap_tbl["MinAbsGap_days"] = np.nan

    for p, grp in gap_tbl.groupby("Patient"):
        eeg_dates = np.asarray(
            eeg_tbl.loc[eeg_tbl["Patient"] == p, "EEG_Date"],
            dtype="datetime64[ns]"
        )
        
        if len(eeg_dates) == 0:
            continue

        for idx in grp.index:
            vd = np.datetime64(gap_tbl.at[idx, "VisitDate"])

            diffs = np.abs(eeg_dates - vd)
            gap_tbl.at[idx, "MinAbsGap_days"] = diffs.min() / np.timedelta64(1, "D")

            gap_tbl["MinAbsGap_years"] = gap_tbl["MinAbsGap_days"] / 365.25
            return gap_tbl
        
    return gap_tbl

def restrict_visits_by_min_abs_gap(vuniq, report_kept, min_days, max_days):
    eeg_raw = report_kept["start_time_deid"].str.strip()
    eeg_dt  = pd.to_datetime(eeg_raw, format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    pid     = report_kept["Patient"].values
    eeg_tbl = pd.DataFrame({"Patient": pid, "EEG_Date": eeg_dt.values})
    eeg_tbl = eeg_tbl[~pd.isna(eeg_tbl["EEG_Date"]) & np.isfinite(pid.astype(float))]

    keep = pd.Series(False, index=vuniq.index)
    for p, grp in vuniq.groupby("Patient"):
        eeg_dates = eeg_tbl.loc[eeg_tbl["Patient"] == p, "EEG_Date"].values
        if len(eeg_dates) == 0:
            continue
        for idx in grp.index:
            vd = grp.at[idx, "VisitDate"]
            min_gap = min(abs((vd - ed).days) for ed in eeg_dates)
            if min_days <= min_gap <= max_days:
                keep[idx] = True

    result = vuniq[keep].copy()
    print(f"[Visit-EEG distance] Kept {keep.sum()}/{len(vuniq)} visits with "
          f"min|gap| in [{min_days}, {max_days}] days")
    return result


# ─────────────────────────────────────────────
# NEAR/FAR TERTILE ANALYSIS
# ─────────────────────────────────────────────

def spearman_rho(x: Any, y: Any) -> float:
    res: Any = stats.spearmanr(x, y)
    rho = cast(float, res[0])
    return rho

def plot_delta_rho_histogram(views, vuniq, report_kept, near_q=0.333, far_q=0.667,
                             n_boot=5000, alpha=0.05, out_png=""):
    base_patients = views["PatientSpikeSz_All"]["Patient"].unique()
    spike_tbl = views["PatientLevelSpikeRates"][["Patient", "MeanSpikeRate_perHour"]].copy()
    spike_tbl = spike_tbl[spike_tbl["Patient"].isin(base_patients)]

    gap_tbl = compute_visit_eeg_gaps(vuniq, report_kept)
    gaps    = gap_tbl["MinAbsGap_days"].dropna().values

    near_days = np.quantile(gaps, near_q)
    far_days  = np.quantile(gaps, far_q)

    v_near = restrict_visits_by_min_abs_gap(vuniq, report_kept, 0,         near_days)
    v_far  = restrict_visits_by_min_abs_gap(vuniq, report_kept, far_days, np.inf)

    vn = v_near[v_near["Patient"].isin(base_patients) & np.isfinite(v_near["Freq_R1"].values)]
    vf = v_far[ v_far["Patient"].isin(base_patients)  & np.isfinite(v_far["Freq_R1"].values)]

    sz_near = build_patient_seizure_metrics(vn).rename(columns={"MeanSzFreq": "MeanSzFreq_near"})
    sz_far  = build_patient_seizure_metrics(vf).rename(columns={"MeanSzFreq": "MeanSzFreq_far"})

    j_near = spike_tbl.merge(sz_near[["Patient", "MeanSzFreq_near"]], on="Patient")
    j_far  = spike_tbl.merge(sz_far[["Patient",  "MeanSzFreq_far"]],  on="Patient")
    J = j_near.merge(j_far[["Patient", "MeanSzFreq_far"]], on="Patient")

    mask = (
        np.isfinite(J["MeanSpikeRate_perHour"].values) &
        np.isfinite(J["MeanSzFreq_near"].values) &
        np.isfinite(J["MeanSzFreq_far"].values)
    )
    J = J[mask].reset_index(drop=True)
    n = len(J)
    if n < 3:
        raise ValueError(f"Not enough patients ({n}) for near/far analysis.")

    x  = J["MeanSpikeRate_perHour"].values
    yN = J["MeanSzFreq_near"].values
    yF = J["MeanSzFreq_far"].values

    rho_near = spearman_rho(x, yN)
    rho_far  = spearman_rho(x, yF)

    delta_obs = rho_near - rho_far

    delta = np.array([
        spearman_rho(x[idx := np.random.randint(0, n, n)], yN[idx]) -
        spearman_rho(x[idx], yF[idx])
        for _ in range(n_boot)
    ])
    ci_lo      = np.percentile(delta, 100 * alpha / 2)
    ci_hi      = np.percentile(delta, 100 * (1 - alpha / 2))
    delta_med  = np.median(delta)
    p_one      = np.mean(delta <= 0)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.patch.set_facecolor("white")

    ax1 = axes[0]
    ax1.hist(gaps, bins=60, color="gray", alpha=0.6, edgecolor="none")
    yl = ax1.get_ylim()
    ax1.fill_betweenx(yl, 0,         near_days, alpha=0.12, color="black")
    ax1.fill_betweenx(yl, near_days, far_days,  alpha=0.06, color="black")
    ax1.fill_betweenx(yl, far_days,  gaps.max(), alpha=0.12, color="black")
    ax1.axvline(near_days, linestyle="--", color="black", linewidth=2)
    ax1.axvline(far_days,  linestyle="--", color="black", linewidth=2)
    ax1.set_xlabel("|Visit - EEG| gap (days)", fontsize=14)
    ax1.set_ylabel("Visit count", fontsize=14)
    ax1.set_title("A. Visit–EEG gap distribution with lower and upper third cutoffs", fontsize=14)
    ax1.text(near_days / 2,   yl[1] * 0.95, "Short gap\n(lower third)", ha="center", va="top", fontsize=12)
    ax1.text((gaps.max() + far_days) / 2, yl[1] * 0.95, "Long gap\n(upper third)", ha="center", va="top", fontsize=12)
    ax1.grid(True, alpha=0.3); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2 = axes[1]
    ax2.hist(delta, bins=40, color="steelblue", alpha=0.7, edgecolor="none")
    ax2.axvline(0,         linestyle="--", color="black", linewidth=2)
    ax2.axvline(delta_med, linestyle="-",  color="black", linewidth=2)
    max_abs = max(abs(delta.min()), abs(delta.max())) * 1.08
    ax2.set_xlim([-max_abs, max_abs])
    ax2.set_xlabel("Δρ = ρ_short_gap − ρ_long_gap", fontsize=14)
    ax2.set_ylabel("Bootstrap count", fontsize=14)
    ax2.set_title(
        f"B. Δρ distribution\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}], p = {p_one:.3g}",
        fontsize=14
    )
    ax2.grid(True, alpha=0.3); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    if out_png:
        save_fig(fig, out_png)

    stats_out = {
        "nPatients": n,
        "nearQ": near_q, "farQ": far_q,
        "nearDays": near_days, "farDays": far_days,
        "rho_near": rho_near, "rho_far": rho_far,
        "delta_obs": delta_obs, "delta_boot": delta,
        "delta_median": delta_med, "delta_ci_lo": ci_lo, "delta_ci_hi": ci_hi,
        "p_one_sided": p_one, "p_two_sided": min(2 * min(p_one, 1 - p_one), 1),
    }
    print(f"\nFig S3 analysis:\nN patients: {n}\n"
          f"rho short gap: {rho_near:.2f}\nrho long gap: {rho_far:.2f}\n"
          f"Median [95% CI] Δρ: {delta_obs:.3f} [{ci_lo:.2f}-{ci_hi:.2f}]\np = {p_one:.4f}")
    return stats_out


# ─────────────────────────────────────────────
# FIGURE S2
# ─────────────────────────────────────────────

def make_figS2(views, sz_freq_per_patient, n_boot=N_BOOT, alpha=ALPHA, x_lims=SPEARMAN_XLIMS):
    rs = resolve_reported_spike_status(views["ReportForKeptSessions"])
    has_present = rs.groupby("Patient")["ReportStatus"].apply(lambda x: any(x == "present"))
    has_absent  = rs.groupby("Patient")["ReportStatus"].apply(lambda x: any(x == "absent"))
    rpt = pd.DataFrame({"HasPresent": has_present, "HasAbsent": has_absent}).reset_index()

    s2 = sz_freq_per_patient.merge(rpt, on="Patient")
    ep_pats = views["PatientLevelSpikeRates"].loc[views["IsEpilepsyMask"], "Patient"].values
    s2 = s2[s2["Patient"].isin(ep_pats)]

    freq_all_absent  = s2.loc[s2["HasAbsent"]  & ~s2["HasPresent"], "MeanSzFreq"].dropna().values
    freq_any_present = s2.loc[s2["HasPresent"],                      "MeanSzFreq"].dropna().values

    _, p = stats.mannwhitneyu(freq_all_absent, freq_any_present, alternative="two-sided")

    EPS_FREQ = 1e-3; y_zero_s2 = np.log10(EPS_FREQ)
    ya = to_log10_per_month(freq_all_absent,  EPS_FREQ)
    yp = to_log10_per_month(freq_any_present, EPS_FREQ)
    ya = add_y_jitter_eps(ya, y_zero_s2, x_lims, 0.02)
    yp = add_y_jitter_eps(yp, y_zero_s2, x_lims, 0.02)

    med1, lo1, hi1 = bootstrap_median_ci(freq_all_absent,  n_boot, alpha)
    med2, lo2, hi2 = bootstrap_median_ci(freq_any_present, n_boot, alpha)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    for xi, (y_g, col, lab, med, lo, hi) in enumerate(zip(
        [ya, yp], ["#4878CF", "#D65F5F"],
        [f"All EEGs: no spikes (N={len(freq_all_absent)})",
         f"≥1 EEG: spikes present (N={len(freq_any_present)})"],
        [med1, med2], [lo1, lo2], [hi1, hi2]
    )):
        bp = ax.boxplot(y_g, positions=[xi + 1], widths=0.4, patch_artist=True,
                        showfliers=False, medianprops=dict(color="k"))
        for patch in bp["boxes"]:
            patch.set_facecolor(col); patch.set_alpha(0.3)
        jitter = np.random.uniform(-0.15, 0.15, len(y_g))
        ax.scatter(xi + 1 + jitter, y_g, alpha=0.2, s=18, color=col)
        # CI overlay for seizures/month
        y_med_log = to_log10_per_month(np.atleast_1d(med), EPS_FREQ)[0]
        y_lo_log  = to_log10_per_month(np.atleast_1d(lo),  EPS_FREQ)[0]
        y_hi_log  = to_log10_per_month(np.atleast_1d(hi),  EPS_FREQ)[0]
        ax.plot([xi + 1, xi + 1], [y_lo_log, y_hi_log], "k-", linewidth=3)
        ax.plot(xi + 1, y_med_log, "ko", markersize=6, markerfacecolor="k")

    ax.axhline(y_zero_s2, linestyle=":", color="gray", linewidth=1.2)
    ax.set_ylim(x_lims)
    set_log10_ticks(ax, "y", EPS_FREQ, x_lims)
    ax.set_ylabel("Seizures/month (log scale)", fontsize=14)
    yl = ax.get_ylim()
    y_max = max(np.concatenate([ya, yp]))
    y_bar = y_max + 0.06 * (yl[1] - yl[0])
    add_sigbar(ax, 1, 2, y_bar, p_label(p))
    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        f"All EEGs: no spikes\n(N={len(freq_all_absent)})",
        f"≥1 EEG: spikes present\n(N={len(freq_any_present)})"
    ], rotation=15, fontsize=11)
    ax.set_title("Mean seizure frequency by reported spikes across EEGs", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    # ── Load ──
    spike_df, report_df = load_data()

    # ── Filter visit arrays ──
    report_df = filter_visit_arrays_by_type(report_df, ALLOWABLE_VISITS)

    # ── Filter outpatient routine ──
    spike_df, report_df, n_patients_total = filter_outpatient_routine(
        spike_df, report_df, DUR_COL, MAX_ROUTINE_HOURS
    )

    # ── Build cohort tables ──
    vuniq            = build_visit_level_table(report_df)
    patient_typing   = build_patient_typing(report_df, CANONICAL3)
    sz_freq_per_pat  = build_patient_seizure_metrics(vuniq)

    views = build_filtered_view(
        spike_df, report_df, patient_typing, sz_freq_per_pat,
        NESD_LABEL, BAD_TYPES, CANONICAL3, n_patients_total
    )

    # ── EEG-visit pair table ──
    pair_table = build_eeg_visit_pairs(
        vuniq,
        views["SessionLevelSpikeRates"],
        views["ReportForKeptSessions"],
        views["PatientTypingFiltered"],
    )
    print(f"Canonical-subtype pairs: {len(pair_table)}, "
          f"patients: {pair_table['Patient'].nunique()}")

    # ── Mixed effects models ──
    mmr = fit_mixed_effects_models(pair_table, N_BOOT, ALPHA)

    # ── Figure 1 ──
    fig1, fig1_stats = make_fig1_controls(views)
    save_fig(fig1, OUTPUT_DIR / "Fig1.png")
    plt.close(fig1)

    # ── Figure 2 (Spearman main) ──
    (spearman_main, rs_all_main, p_all_main, n_all_main,
     rho_lo_main, rho_hi_main, subtype_ci_main) = spearman_plotting_function(
        views["PatientSpikeSz_All"], views["PatientSpikeSz_Typed"],
        CANONICAL3, SPEARMAN_XLIMS, SPEARMAN_YLIMS,
        OUTPUT_DIR / "Fig2.png", "MeanSzFreq", "", False
    )

    # ── Figure S1 (Spearman, non-zero only) ──
    (spearman_s1, rs_all_s1, p_all_s1, n_all_s1,
     rho_lo_s1, rho_hi_s1, subtype_ci_s1) = spearman_plotting_function(
        views["PatientSpikeSz_All"], views["PatientSpikeSz_Typed"],
        CANONICAL3, SPEARMAN_XLIMS, SPEARMAN_YLIMS,
        OUTPUT_DIR / "FigS1.png", "MeanSzFreq", " (positive spike/seizures only)", True
    )

    # ── Figure S2 ──
    figs2 = make_figS2(views, sz_freq_per_pat)
    save_fig(figs2, OUTPUT_DIR / "FigS2.png")
    plt.close(figs2)

    # ── Near/far tertile figure (S_Tertile) ──
    near_far_stats = plot_delta_rho_histogram(
        views, vuniq, views["ReportForKeptSessions"],
        0.333, 0.667, N_BOOT, ALPHA,
        str(OUTPUT_DIR / "FigSTertile.png")
    )

    # ── Table 1 ──
    table1 = build_table1_flat(views, sz_freq_per_pat, vuniq)
    table1.to_csv(OUTPUT_DIR / "Table1.csv", index=False)
    print(f"Wrote Table 1: {OUTPUT_DIR / 'Table1.csv'}")

    print("\nPipeline complete. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
