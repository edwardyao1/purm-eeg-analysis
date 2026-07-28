import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Helpers ──────────────────────────────────────────────────────────────────
def track_filter(label, before, after):
    removed = before - after
    pct = removed / before * 100 if before > 0 else 0
    print(f"  {label:55s}  {before:5d} → {after:5d}  (removed {removed:4d},  {pct:.1f}%)")

def parse_sz_freq(val):
    """
    Safely converts sz_freqs strings like '[2.5]' or '[1.0, null, 2.0]' into a single numeric float.
    Takes the mean if multiple valid numbers are present in the list.
    """
    if pd.isna(val): 
        return np.nan
    
    val_str = str(val).strip().strip("[]").replace("'", "").replace('"', "")
    if not val_str: 
        return np.nan
    
    try:
        parts = [float(x.strip()) for x in val_str.split(",") if x.strip() and x.strip().lower() != 'null']
        if not parts:
            return np.nan
        return np.mean(parts)
    except ValueError:
        return np.nan

# def canonical_subtype(row):
#     epi  = str(row.get("epilepsy_type", "")).strip()
#     spec = str(row.get("epilepsy_specific", "")).strip()
#     jfoc = str(row.get("jay_focal_epi", "")).strip().lower()
#     jmul = str(row.get("jay_multifocal_epi", "")).strip().lower()

#     if epi == "Focal":
#         if "temporal" in spec.lower(): return "Temporal Lobe"
#         if "frontal" in spec.lower(): return "Frontal Lobe"
#         if "parietal" in spec.lower() or "occipital" in spec.lower(): return "Parietal/Occipital"
#         if jmul == "present": return "Multifocal"
#         return "Unlocalized Focal"
#     if epi == "General":
#         if "jme" in spec.lower() or "myoclonic" in spec.lower(): return "JME"
#         if "gtca" in spec.lower() or "tonic-clonic" in spec.lower(): return "GTCA"
#         return "Unspecified Generalized"
#     return epi


# ── 1. Load & Merge ────────────────────────────────────────────────────────
print("=" * 75)
print("LOADING DATA")
print("=" * 75)

spike_df    = pd.read_csv("/Users/edwardyao/Documents/PURM/data/spike_counts.csv")
clinical_df = pd.read_csv("/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv")
clinical_df = clinical_df.rename(columns={"patient_id": "Patient", "session_number": "Session"})

print(f"  spike_counts:           {len(spike_df):6,} sessions")
print(f"  clinical_data:          {len(clinical_df):6,} records\n")

print("=" * 75)
print("INITIAL MERGE")
print("=" * 75)
df = pd.merge(spike_df, clinical_df, on=["Patient", "Session"], how="outer")
print(f"  Merged dataset:  {len(df):,} total sessions\n")


# ── 2. Filters ─────────────────────────────────────────────────────────────
print("=" * 75)
print("SESSION-LEVEL FILTERS")
print("=" * 75)

allowable_visits = {
    "CONSULT VISIT", "ESTABLISHED PATIENT VISIT", "FOLLOW-UP PATIENT CLINIC",
    "NEW PATIENT CLINIC", "NEW PATIENT VISIT", "NPV MANAGEMENT DURING COVID-19",
    "NPV NEUROLOGY", "RETURN ANNUAL VISIT", "RETURN PATIENT EXTENDED",
    "RETURN PATIENT VISIT", "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN"
}

before = len(df)
df = df[df["visit_type"].astype(str).apply(lambda v: any(x in v for x in allowable_visits))].copy()
track_filter("Allowable visit types", before, len(df))

acq = df["acquired_on"].fillna("").astype(str).str.lower()
patient_class = df["report_PATIENT_CLASS"].fillna("").astype(str).str.lower()
jay = df["jay_in_or_out"].fillna("").astype(str).str.lower()

before = len(df)
df = df[acq.str.contains("spe") | acq.str.contains("radnor") | (patient_class == "outpatient") | (jay == "out")].copy()
track_filter("Acquisition location (SPE/Radnor/outpatient/out)", before, len(df))

before = len(df)
df = df[df["Duration_sec"] > 0].copy()
track_filter("Duration > 0 s", before, len(df))

before = len(df)
df = df[df["Duration_sec"] <= 4 * 3600].copy()
track_filter("Duration <= 4 hours", before, len(df))

before = len(df)
df = df[df["count_0_46"].notna()].copy()
track_filter("Valid count_0_46 (spike count)", before, len(df))

before = len(df)
df = df[df["nlp_gender"].isin(["M", "F"])].copy()
track_filter("Valid gender (M or F)", before, len(df))

before = len(df)
df = df[df["epilepsy_type"].isin(["Focal", "General"])].copy()
track_filter("STRICT ALLOWLIST (Focal/General ONLY)", before, len(df))

df["sz_freq_numeric"] = df["sz_freqs"].apply(parse_sz_freq)
df = df[df["sz_freq_numeric"].notna()].copy()
track_filter("Valid numeric seizure frequency", before, len(df))

# df["canonical_subtype"] = df.apply(canonical_subtype, axis=1)

print(f"\n  Sessions remaining after all filters: {len(df):,}")
print(f"  Epilepsy type breakdown:")
for et, cnt in df["epilepsy_type"].value_counts().items():
    print(f"    {et:40s} {cnt:4d}")


# ── 3. Aggregate ───────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("AGGREGATING TO PATIENT LEVEL")
print("=" * 75)

patient_df = df.groupby(["Patient", "nlp_gender", "epilepsy_type"], dropna=False).agg(
    total_spikes   = ("count_0_46",    "sum"),
    total_duration = ("Duration_sec",  "sum"),
    mean_sz_freq   = ("sz_freq_numeric","mean"),
).reset_index()

patient_df["spike_rate_per_hour"] = (patient_df["total_spikes"] / patient_df["total_duration"]) * 3600

n_before = len(patient_df)
patient_df = patient_df.dropna(subset=["spike_rate_per_hour", "mean_sz_freq"])

print(f"  FINAL PATIENT COHORT SIZE:  {len(patient_df):,}")
print(f"  Breakdown by epilepsy type:")
for et, cnt in patient_df["epilepsy_type"].value_counts().items():
    print(f"    {et:40s} {cnt:4d}")


# ── 4. Bootstrap ───────────────────────────────────────────────────────────
def bootstrap_median_diff(group_a, group_b, n_boot=5000):
    a, b = np.array(group_a), np.array(group_b)
    obs_diff = np.median(b) - np.median(a)
    boot_diffs = np.array([
        np.median(np.random.choice(b, len(b), replace=True)) - np.median(np.random.choice(a, len(a), replace=True))
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5)
    p = 2 * np.mean(boot_diffs <= 0) if obs_diff >= 0 else 2 * np.mean(boot_diffs >= 0)
    return obs_diff, ci_lo, ci_hi, min(p, 1.0)

def run_analysis(patient_df, outcome_col, outcome_label):
    print(f"\n===========================================================================")
    print(f"BOOTSTRAP ANALYSIS — {outcome_label}")
    print(f"===========================================================================")
    rows = []
    
    # Sex
    female = patient_df.loc[patient_df["nlp_gender"] == "F", outcome_col]
    male   = patient_df.loc[patient_df["nlp_gender"] == "M", outcome_col]
    ref_med = np.median(female)
    diff, lo, hi, p = bootstrap_median_diff(female, male)
    
    rows.append(dict(section="Sex", label="Female", n=len(female), ref_med=ref_med, is_reference=True, diff=0, lo=0, hi=0, p=None))
    rows.append(dict(section="Sex", label="Male", n=len(male), ref_med=None, is_reference=False, diff=diff, lo=lo, hi=hi, p=p))

    # Epilepsy Type (Only Focal and General exist now)
    focal   = patient_df.loc[patient_df["epilepsy_type"] == "Focal", outcome_col]
    general = patient_df.loc[patient_df["epilepsy_type"] == "General", outcome_col]
    ref_med_epi = np.median(focal)
    
    rows.append(dict(section="Epilepsy Type", label="Focal", n=len(focal), ref_med=ref_med_epi, is_reference=True, diff=0, lo=0, hi=0, p=None))
    
    if len(general) >= 5:
        diff, lo, hi, p = bootstrap_median_diff(focal, general)
        rows.append(dict(section="Epilepsy Type", label="General", n=len(general), ref_med=None, is_reference=False, diff=diff, lo=lo, hi=hi, p=p))

    return rows

rows_spike = run_analysis(patient_df, "spike_rate_per_hour", "SPIKE RATE (per hour)")
rows_sz    = run_analysis(patient_df, "mean_sz_freq", "SEIZURE FREQUENCY (per month)")

# ── 5. Render ──────────────────────────────────────────────────────────────
def pval_str(p):
    if p is None: return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def forest_plot(rows, title, x_label, x_lim, x_ticks, out_path):
    fig, ax = plt.subplots(figsize=(14, 7)) # Kept wide for horizontal spread
    
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # --- FIX 1: Adjusted Column x-coordinates for better horizontal spread ---
    COL_SUBGROUP = 0.01
    COL_N = 0.20
    COL_EST = 0.35
    COL_PLOT_L = 0.50
    COL_PLOT_R = 0.88
    COL_P = 0.92
    plot_w = COL_PLOT_R - COL_PLOT_L

    def data_to_x(val):
        frac = (val - x_lim[0]) / (x_lim[1] - x_lim[0])
        return COL_PLOT_L + frac * plot_w

    # --- Robust Vertical Position (Y) Mapping ---
    sections = []
    for r in rows:
        if r["section"] not in sections:
            sections.append(r["section"])

    ypos = {}
    y_val = len(rows) * 1.5 + len(sections) * 2.0 
    
    for sec in sections:
        ypos[(sec, "__header__")] = y_val
        y_val -= 1.3 
        
        for r in rows:
            if r["section"] == sec:
                ypos[(sec, r["label"])] = y_val
                y_val -= 1.3 
                
        y_val -= 1.0 

    top_y = max(ypos.values())
    bottom_y = min(ypos.values())
    
    Y_TOP_RULE = top_y + 1.0
    Y_HEADER = Y_TOP_RULE + 0.7
    Y_BOTTOM_RULE = bottom_y - 0.6
    Y_XAXIS = Y_BOTTOM_RULE - 1.2

    ax.set_xlim(0, 1)
    ax.set_ylim(Y_XAXIS - 2.5, Y_HEADER + 0.5)

    # --- Draw Headers & Lines ---
    hkw = dict(ha="left", va="center", fontsize=11, fontweight="bold", color="#111111")
    ax.text(COL_SUBGROUP, Y_HEADER, "Subgroup", **hkw)
    ax.text(COL_N,        Y_HEADER, "No. of Patients", **hkw)
    ax.text(COL_EST,      Y_HEADER, "Coefficient (95% CI)", **hkw)
    ax.text(COL_P,        Y_HEADER, "P-Value", **hkw)

    # Top Border
    ax.plot([0, 1], [Y_TOP_RULE, Y_TOP_RULE], color="black", lw=1.5)
    
    # --- FIX 2: Removed the Bottom Border ---
    # ax.plot([0, 1], [Y_BOTTOM_RULE, Y_BOTTOM_RULE], color="black", lw=1.5) 

    # Reference dashed line
    ax.plot([data_to_x(0), data_to_x(0)], [Y_TOP_RULE, Y_XAXIS], color="gray", lw=1.0, ls="--", zorder=1)

    # X-Axis & Ticks
    ax.plot([data_to_x(x_lim[0]), data_to_x(x_lim[1])], [Y_XAXIS, Y_XAXIS], color="black", lw=1.2)
    for tx in x_ticks:
        x_pos = data_to_x(tx)
        ax.plot([x_pos, x_pos], [Y_XAXIS, Y_XAXIS - 0.15], color="black", lw=1.0)
        ax.text(x_pos, Y_XAXIS - 0.40, str(tx), ha="center", va="top", fontsize=10)

    # Axis Labels
    ax.text((COL_PLOT_L + COL_PLOT_R)/2, Y_XAXIS - 1.2, x_label, ha="center", fontsize=11, fontweight="bold")

    # Title
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.94)

    # --- Render the Data Text and Markers ---
    for sec in sections:
        ax.text(COL_SUBGROUP, ypos[(sec, "__header__")], sec, fontsize=11, fontweight="bold")

    for row in rows:
        y_pos = ypos[(row["section"], row["label"])]
        
        ax.text(COL_SUBGROUP + 0.04, y_pos, row["label"], fontsize=10)
        ax.text(COL_N, y_pos, str(row["n"]), fontsize=10)

        if row["is_reference"]:
            ax.text(COL_EST, y_pos, f"Reference", fontsize=10)
            ax.plot(data_to_x(0), y_pos, marker="s", color="black", ms=7)
            
            ax.text(COL_P, y_pos, "Ref.", fontsize=10)
            
        else:
            d, lo, hi = row["diff"], row["lo"], row["hi"]
            ax.text(COL_EST, y_pos, f"{d:+.2f} ({lo:+.2f}, {hi:+.2f})", fontsize=10)
            
            # Confidence interval lines
            ax.plot([data_to_x(lo), data_to_x(hi)], [y_pos, y_pos], color="black", lw=2)
            # Estimate square marker
            ax.plot(data_to_x(d), y_pos, marker="s", color="black", ms=7)
            
            ax.text(COL_P, y_pos, pval_str(row["p"]), fontsize=10)

    plt.subplots_adjust(top=0.88, bottom=0.1)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# # Note: Ensure the function calls match the parameters you need, like this:
# forest_plot(
#     rows_spike,
#     "Bootstrapped OLS Coefficients: Spike Rate (per hour)",
#     "Coefficient",
#     (-0.4, 0.8),
#     [-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
#     "/Users/edwardyao/Documents/PURM/data/forest_spike_rate_median.png"
# )

# forest_plot(
#     rows_sz,
#     "Bootstrapped OLS Coefficients: Seizure Frequency (per month)",
#     "Coefficient",
#     (-0.2, 0.5),
#     [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
#     "/Users/edwardyao/Documents/PURM/data/forest_seizure_freq_median.png"
# )
print("\nDone.")

forest_plot(
    rows_spike, 
    "Differences in Median Spike Rate (per hour)", 
    "Difference in Medians (Linear Scale)", 
    (-1.0, 2.5), [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], 
    "/Users/edwardyao/Documents/PURM/gender_project_output/forest_spike_rate_median.png"
    )

forest_plot(
    rows_sz, 
    "Differences in Median Seizure Frequency (per month)", 
    "Difference in Medians (Linear Scale)", 
    (-2, 5), [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 
    "/Users/edwardyao/Documents/PURM/gender_project_output/forest_seizure_freq_median.png"
    )

print("\nDone.")