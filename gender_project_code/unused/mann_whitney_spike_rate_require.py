import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def track_filter(step_name, before, after):
    removed = before - after
    print(f"--- {step_name} ---")
    print(f"Started with {before} rows, removed {removed}. Remaining: {after}\n")

# ==========================================================
# PARSERS & HELPERS
# ==========================================================
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

def mwu_effect_size(df, group_col, value_col, g1="M", g2="F"):
    x = df[df[group_col] == g1][value_col].dropna()
    y = df[df[group_col] == g2][value_col].dropna()

    n1, n2 = len(x), len(y)

    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, n1, n2

    u, p = mannwhitneyu(x, y, alternative="two-sided")
    effect = 1 - (2 * u) / (n1 * n2)

    return u, p, effect, n1, n2

def canonical_subtype(row):
    etype = str(row.get("epilepsy_type", "")).lower()
    espec = str(row.get("epilepsy_specific", "")).lower()

    if "temporal" in espec: return "Temporal"
    if "frontal" in espec: return "Frontal"
    if "focal" in etype: return "Focal"
    if etype == "general": return "General"
    
    return np.nan

# ==========================================================
# REUSABLE ANALYSIS FUNCTION
# ==========================================================
def run_statistical_analysis(patient_df, target_var, target_name):
    print(f"\n======================================================")
    print(f"=== MANN-WHITNEY U: {target_name.upper()} ===")
    print(f"======================================================\n")

    # 1. PRIMARY COHORT
    print("=== PRIMARY: OVERALL COHORT ===")
    u, p, es, n1, n2 = mwu_effect_size(patient_df, "nlp_gender", target_var)
    print(f"M = {n1}, F = {n2}")
    if n1 > 0 and n2 > 0:
        print(f"U = {u:.2f}, p = {p:.6f}, effect = {es:.4f}")
    else:
        print("Insufficient data for analysis.")

    # 2. EPILEPSY TYPE
    print("\n=== EPILEPSY TYPE ANALYSIS ===")
    type_results = []
    type_pvals = []

    for t in ["Focal", "General"]:
        sub = patient_df[patient_df["epilepsy_type"] == t]
        u, p, es, n1, n2 = mwu_effect_size(sub, "nlp_gender", target_var)
        type_results.append((t, u, p, es, n1, n2))
        type_pvals.append(1.0 if pd.isna(p) else p)

    reject, p_adj, _, _ = multipletests(type_pvals, method="bonferroni")

    for (t, u, p, es, n1, n2), pa, sig in zip(type_results, p_adj, reject):
        print(f"\nType: {t}")
        print(f"M = {n1}, F = {n2}")
        if n1 == 0 or n2 == 0:
            print("Insufficient data.")
        else:
            print(f"U = {u:.2f} | Raw p = {p:.6f} | Bonf p = {pa:.6f} | Effect = {es:.4f} | Sig = {sig}")

    # 3. FOCAL SUBTYPES
    print("\n=== FOCAL SUBTYPES ===")
    focal_df = patient_df[patient_df["epilepsy_type"] == "Focal"].copy()
    focal_df = focal_df[focal_df["canonical_subtype"].isin(["Temporal", "Frontal"])]
    
    focal_results = []
    focal_pvals = []

    for st in ["Temporal", "Frontal"]:
        sub = focal_df[focal_df["canonical_subtype"] == st]
        u, p, es, n1, n2 = mwu_effect_size(sub, "nlp_gender", target_var)
        focal_results.append((st, u, p, es, n1, n2))
        focal_pvals.append(1.0 if pd.isna(p) else p)

    reject, p_adj, _, _ = multipletests(focal_pvals, method="bonferroni")

    for (st, u, p, es, n1, n2), pa, sig in zip(focal_results, p_adj, reject):
        print(f"\nSubtype: {st}")
        print(f"M = {n1}, F = {n2}")
        if n1 == 0 or n2 == 0:
            print("Insufficient data.")
        else:
            print(f"U = {u:.2f} | Raw p = {p:.6f} | Bonf p = {pa:.6f} | Effect = {es:.4f} | Sig = {sig}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    # ----------------------------
    # LOAD
    # ----------------------------
    clinical_df = pd.read_csv('/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv')
    spike_df = pd.read_csv('/Users/edwardyao/Documents/PURM/data/spike_counts.csv')

    # ----------------------------
    # 1. INITIAL MERGE (OUTER JOIN)
    # ----------------------------
    clinical_df = clinical_df.rename(columns={'patient_id': 'Patient', 'session_number': 'Session'})
    df = pd.merge(spike_df, clinical_df, on=["Patient", "Session"], how="outer")
    print(f"--- Initial Merge ---\nMerged dataset starts with {len(df)} total sessions\n")

    # ----------------------------
    # 2. APPLY SESSION FILTERS
    # ----------------------------
    allowable_visits = {
        "CONSULT VISIT", "ESTABLISHED PATIENT VISIT", "FOLLOW-UP PATIENT CLINIC",
        "NEW PATIENT CLINIC", "NEW PATIENT VISIT", "NPV MANAGEMENT DURING COVID-19",
        "NPV NEUROLOGY", "RETURN ANNUAL VISIT", "RETURN PATIENT EXTENDED",
        "RETURN PATIENT VISIT", "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN"
    }

    before = len(df)
    df = df[df["visit_type"].astype(str).apply(lambda v: any(x in v for x in allowable_visits))].copy()
    track_filter("Filter: Allowable Visits", before, len(df))

    acq = df["acquired_on"].fillna("").astype(str).str.lower()
    patient_class = df["report_PATIENT_CLASS"].fillna("").astype(str).str.lower()
    jay = df["jay_in_or_out"].fillna("").astype(str).str.lower()
    before = len(df)
    df = df[acq.str.contains("spe") | acq.str.contains("radnor") | (patient_class == "outpatient") | (jay == "out")].copy()
    track_filter("Filter: MATLAB Acquisition Locations", before, len(df))

    before = len(df)
    df = df[df["Duration_sec"] > 0].copy()
    track_filter("Filter: Duration > 0", before, len(df))

    before = len(df)
    df = df[df["Duration_sec"] <= 4 * 3600].copy()
    track_filter("Filter: Duration <= 4 hours", before, len(df))
    
    before = len(df)
    df = df[df["nlp_gender"].isin(["M", "F"])].copy()
    track_filter("Filter: Valid Gender (M/F)", before, len(df))
    
    bad_types = {"Uncertain if Epilepsy", "Unknown or MRN not found", "", "Non-Epileptic Seizure Disorder", "Unclassified or Unspecified"}
    before = len(df)
    df = df[~df["epilepsy_type"].isin(bad_types)].copy()
    track_filter("Filter: Remove Bad Epilepsy Types", before, len(df))

    before = len(df)
    df["sz_freq_numeric"] = df["sz_freqs"].apply(parse_sz_freq)
    df = df[df["sz_freq_numeric"].notna()].copy()
    track_filter("Filter: Valid Numeric Seizure Frequency", before, len(df))

    df["canonical_subtype"] = df.apply(canonical_subtype, axis=1)

    # ----------------------------
    # 3. AGGREGATE TO PATIENT LEVEL
    # ----------------------------
    patient_df = df.groupby(
        ["Patient", "nlp_gender", "epilepsy_type", "canonical_subtype"], 
        dropna=False
    ).agg(
        total_duration=("Duration_sec", "sum"),
        mean_sz_freq=("sz_freq_numeric", "mean") 
    ).reset_index()

    # Calculate spike rate per hour using merged session durations

    # Strict Validation: Patient must have BOTH metrics to exist in the final cohort
    before_drop = len(patient_df)
    patient_df = patient_df.dropna(subset=["mean_sz_freq"])

    print("--- Aggregating to Patient Level ---")
    print(f"Grouped {len(df)} valid sessions into {before_drop} patients.")
    print(f"Removed {before_drop - len(patient_df)} patients missing final aggregated metrics.")
    print(f"FINAL PATIENT COHORT SIZE: {len(patient_df)}\n")

    # ----------------------------
    # 4. RUN STATISTICS
    # ----------------------------

    # Run test for Seizure Frequency
    run_statistical_analysis(patient_df, target_var="mean_sz_freq", target_name="Average Seizure Frequency")


if __name__ == "__main__":
    main()