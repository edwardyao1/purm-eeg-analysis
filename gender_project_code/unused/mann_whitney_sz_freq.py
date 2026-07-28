import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def track_filter(step_name, before, after):
    removed = before - after
    print(f"--- {step_name} ---")
    print(f"started with {before} amount of sample size then removed {removed} much now we have {after} much\n")

# ==========================================================
# STATISTICS
# ==========================================================
# Mann-Whitney U test with effect size calculation for non-parametric data
def mwu_effect_size(df, group_col, value_col, g1="M", g2="F"):
    # Drop NaNs for the specific group comparisons
    x = df[df[group_col] == g1][value_col].dropna()
    y = df[df[group_col] == g2][value_col].dropna()

    n1, n2 = len(x), len(y)

    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, n1, n2

    # Mann-Whitney U test
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    
    # Effect size (r) calculation: r = 1 - (2U / (n1*n2))
    effect = 1 - (2 * u) / (n1 * n2)

    return u, p, effect, n1, n2


# ==========================================================
# CANONICAL SUBTYPE
# ==========================================================
# Define canonical subtype based on epilepsy_specific and epilepsy_type fields
def canonical_subtype(row):
    etype = str(row.get("epilepsy_type", "")).lower()
    espec = str(row.get("epilepsy_specific", "")).lower()

    if "temporal" in espec:
        return "Temporal"
    if "frontal" in espec:
        return "Frontal"
    if "focal" in etype:
        return "Focal"
    if etype == "general":
        return "General"
    
    return np.nan


# ==========================================================
# MAIN
# ==========================================================
def main():

    # ----------------------------
    # LOAD
    # ----------------------------
    clinical_df = pd.read_csv('/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv')
    spike_df = pd.read_csv('/Users/edwardyao/Documents/PURM/data/spike_counts.csv')

    # ==========================================================
    # 1. INITIAL MERGE (OUTER JOIN)
    # ==========================================================
    # Rename clinical columns to match spike data for a cleaner merge
    clinical_df = clinical_df.rename(columns={'patient_id': 'Patient', 'session_number': 'Session'})
    
    # Outer join to keep ALL sessions initially from both datasets. 
    # Matches will be merged into the same row. Unmatched will have NaNs in the missing columns.
    df = pd.merge(spike_df, clinical_df, on=["Patient", "Session"], how="outer")
    
    print(f"--- Initial Merge ---")
    print(f"Merged dataset starts with {len(df)} total sessions (including unmatched)\n")

    # ==========================================================
    # 2. APPLY ALL FILTERS TO THE MERGED DATASET
    # ==========================================================

    allowable_visits = {
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
    }

    # Filter 1: Allowable visits
    before_len = len(df)
    df = df[df["visit_type"].astype(str).apply(lambda v: any(x in v for x in allowable_visits))].copy()
    track_filter("Filter: Allowable Visits", before_len, len(df))

    # Filter 2: MATLAB Acquisition Locations & Patient Class
    acq = df["acquired_on"].fillna("").astype(str).str.lower()
    patient_class = df["report_PATIENT_CLASS"].fillna("").astype(str).str.lower()
    jay = df["jay_in_or_out"].fillna("").astype(str).str.lower()

    before_len = len(df)
    df = df[
        acq.str.contains("spe") |
        acq.str.contains("radnor") |
        (patient_class == "outpatient") |
        (jay == "out")
    ].copy()
    track_filter("Filter: MATLAB Acquisition Locations", before_len, len(df))

    # Filter 3: Recording Duration > 0 (ALLOW NaNs for those without spike data)
    before_len = len(df)
    df = df[(df["Duration_sec"] > 0) | (df["Duration_sec"].isna())].copy()
    track_filter("Filter: Duration > 0 (or missing)", before_len, len(df))

    # Filter 4: Recording Duration <= 4 hours (ALLOW NaNs for those without spike data)
    before_len = len(df)
    df = df[(df["Duration_sec"] <= 4 * 3600) | (df["Duration_sec"].isna())].copy()
    track_filter("Filter: Duration <= 4 hours (or missing)", before_len, len(df))

    # Filter 5 REMOVED: We no longer require a valid count_0_46

    # Filter 6: Valid Gender (M/F)
    before_len = len(df)
    df = df[df["nlp_gender"].isin(["M", "F"])].copy()
    track_filter("Filter: Valid Gender (M/F)", before_len, len(df))
    
    # Filter 7: Remove Bad Epilepsy Types
    bad_types = {
        "Uncertain if Epilepsy",
        "Unknown or MRN not found",
        "",
        "Non-Epileptic Seizure Disorder",
        "Unclassified or Unspecified"
    }
    before_len = len(df)
    df = df[~df["epilepsy_type"].isin(bad_types)].copy()
    track_filter("Filter: Remove Bad Epilepsy Types", before_len, len(df))

    # Assign Canonical Subtypes
    df["canonical_subtype"] = df.apply(canonical_subtype, axis=1)

    # ==========================================================
    # 3. AGGREGATE TO PATIENT LEVEL
    # ==========================================================
    # Group by patient to calculate total spikes and duration.
    # We use min_count=1 so that if a patient has NO spike data, the sum returns NaN instead of 0.
    patient_df = df.groupby(
        ["Patient", "nlp_gender", "epilepsy_type", "canonical_subtype"], 
        dropna=False
    ).agg(
        total_spikes=("count_0_46", lambda x: x.sum(min_count=1)),
        total_duration=("Duration_sec", lambda x: x.sum(min_count=1))
    ).reset_index()

    # Calculate spike rate per hour for each patient (Will be NaN if they had no spike data)
    patient_df["spike_rate_per_hour"] = (
        patient_df["total_spikes"] / patient_df["total_duration"] * 3600
    )
    
    print("--- Aggregating to Patient Level ---")
    print(f"started with {len(df)} amount of sample size then removed {len(df) - len(patient_df)} much now we have {len(patient_df)} much\n")

    # ==========================================================
    # 4. PRIMARY ANALYSIS (INCLUDES ALL EPILEPSY PATIENTS)
    # ==========================================================
    print("\n=== PRIMARY: OVERALL COHORT ===")
    u, p, es, n1, n2 = mwu_effect_size(patient_df, "nlp_gender", "spike_rate_per_hour")

    print(f"M = {n1}, F = {n2}")
    print(f"U = {u:.2f}, p = {p:.6f}, effect = {es:.4f}")

    # ==========================================================
    # 5. EPILEPSY TYPE (Focal vs General)
    # ==========================================================
    print("\n=== EPILEPSY TYPE ANALYSIS ===")

    type_results = []
    type_pvals = []

    for t in ["Focal", "General"]:
        before_len = len(patient_df)
        sub = patient_df[patient_df["epilepsy_type"] == t]
        track_filter(f"Filter Subgroup: Epilepsy Type == {t}", before_len, len(sub))
        
        u, p, es, n1, n2 = mwu_effect_size(sub, "nlp_gender", "spike_rate_per_hour")
        type_results.append((t, u, p, es, n1, n2))
        type_pvals.append(p)

    reject, p_adj, _, _ = multipletests(type_pvals, method="bonferroni")

    for (t, u, p, es, n1, n2), pa, sig in zip(type_results, p_adj, reject):
        print(f"\nType: {t}")
        print(f"M = {n1}, F = {n2}")
        print(f"U = {u:.2f}")
        print(f"Raw p = {p:.6f}")
        print(f"Bonferroni p = {pa:.6f}")
        print(f"Effect = {es:.4f}")
        print(f"Significant = {sig}")

    # ==========================================================
    # 6. FOCAL SUBTYPE (STRICT: ONLY TEMPORAL/FRONTAL)
    # ==========================================================
    print("\n=== FOCAL SUBTYPES ===")

    before_len = len(patient_df)
    focal_df = patient_df[patient_df["epilepsy_type"] == "Focal"].copy()
    track_filter("Filter Focal Subtype Analysis: Focal only", before_len, len(focal_df))

    before_len = len(focal_df)
    focal_df = focal_df[focal_df["canonical_subtype"].isin(["Temporal", "Frontal"])].copy()
    track_filter("Filter Focal Subtype Analysis: Temporal/Frontal only", before_len, len(focal_df))

    focal_results = []
    focal_pvals = []

    for st in ["Temporal", "Frontal"]:
        before_len = len(focal_df)
        sub = focal_df[focal_df["canonical_subtype"] == st]
        track_filter(f"Filter Subgroup: Canonical Subtype == {st}", before_len, len(sub))

        u, p, es, n1, n2 = mwu_effect_size(sub, "nlp_gender", "spike_rate_per_hour")
        focal_results.append((st, u, p, es, n1, n2))
        focal_pvals.append(p)

    reject, p_adj, _, _ = multipletests(focal_pvals, method="bonferroni")

    for (st, u, p, es, n1, n2), pa, sig in zip(focal_results, p_adj, reject):
        print(f"\nSubtype: {st}")
        print(f"M = {n1}, F = {n2}")
        print(f"U = {u:.2f}")
        print(f"Raw p = {p:.6f}")
        print(f"Bonferroni p = {pa:.6f}")
        print(f"Effect = {es:.4f}")
        print(f"Significant = {sig}")


if __name__ == "__main__":
    main()