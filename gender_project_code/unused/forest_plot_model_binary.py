import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ==========================================================
# 1. HELPERS & PARSERS
# ==========================================================

def track_filter(step_name, before, after):
    removed = before - after
    print(f"--- {step_name} ---")
    print(f"Started with {before} rows, removed {removed}. Remaining: {after}\n")

def parse_sz_freq(val):
    if pd.isna(val): return np.nan
    val_str = str(val).strip().strip("[]").replace("'", "").replace('"', "")
    if not val_str: return np.nan
    try:
        parts = [float(x.strip()) for x in val_str.split(",") if x.strip() and x.strip().lower() != 'null']
        return np.mean(parts) if parts else np.nan
    except ValueError:
        return np.nan

def canonical_subtype(row):
    etype = str(row.get("epilepsy_type", "")).lower()
    if "focal" in etype: return "Focal"
    if etype == "general": return "General"
    return np.nan

# ==========================================================
# 2. CONTINUOUS STATS (For Console Printing)
# ==========================================================

def bootstrap_median_ci(data, n_bootstraps=10000, alpha=0.05):
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    np.random.seed(42) 
    bootstrapped_medians = [np.median(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstraps)]
    lower_bound = np.percentile(bootstrapped_medians, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrapped_medians, 100 * (1 - alpha / 2))
    return np.median(data), lower_bound, upper_bound

def run_cohort_analysis(df, metric, cohort_name):
    print(f"\n======================================================")
    print(f"=== Metric: {metric} | Cohort: {cohort_name} ===")
    print(f"======================================================")
    
    males = df[df['nlp_gender'] == 'M'][metric].dropna().values
    females = df[df['nlp_gender'] == 'F'][metric].dropna().values
    
    if len(males) == 0 or len(females) == 0:
        print("Not enough data in one or both gender groups to compare.\n")
        return

    if len(males) >= len(females):
        ref_group, comp_group = "Male (M)", "Female (F)"
        ref_data, comp_data = males, females
    else:
        ref_group, comp_group = "Female (F)", "Male (M)"
        ref_data, comp_data = females, males
        
    print(f"Reference Group (Largest): {ref_group} (n={len(ref_data)})")
    print(f"Comparison Group:          {comp_group} (n={len(comp_data)})\n")
    
    ref_med, ref_low, ref_up = bootstrap_median_ci(ref_data)
    comp_med, comp_low, comp_up = bootstrap_median_ci(comp_data)
    
    print(f"{ref_group} Median: {ref_med:.4f} (95% CI: {ref_low:.4f} - {ref_up:.4f})")
    print(f"{comp_group} Median: {comp_med:.4f} (95% CI: {comp_low:.4f} - {comp_up:.4f})")
    
    stat, p_val = mannwhitneyu(ref_data, comp_data, alternative='two-sided')
    sig_marker = "*" if p_val < 0.05 else ""
    print(f"\nMann-Whitney U P-value: {p_val:.6f} {sig_marker}")

# ==========================================================
# 3. BINARY LOGISTIC MODELS WITH ADJUSTED P-VALUES
# ==========================================================

def run_logistic_models(df, target_col):
    results = []
    
    # --- SEX MODEL (Ref: Female) ---
    df_sex = df.dropna(subset=['nlp_gender', target_col]).copy()
    if len(df_sex) > 0:
        model_sex = smf.logit(f"{target_col} ~ C(nlp_gender, Treatment('F'))", data=df_sex).fit(disp=0)
        params, conf, pvals = model_sex.params, model_sex.conf_int(), model_sex.pvalues
        
        results.append({
            'category': 'Sex', 'subgroup': 'Female', 'n': len(df_sex[df_sex['nlp_gender'] == 'F']),
            'or': 1.0, 'low': np.nan, 'high': np.nan, 'pval': np.nan, 'is_ref': True
        })
        var_name = "C(nlp_gender, Treatment('F'))[T.M]"
        if var_name in params:
            results.append({
                'category': '', 'subgroup': 'Male', 'n': len(df_sex[df_sex['nlp_gender'] == 'M']),
                'or': np.exp(params[var_name]), 
                'low': np.exp(conf.loc[var_name, 0]), 'high': np.exp(conf.loc[var_name, 1]), 
                'pval': pvals[var_name], 'is_ref': False
            })

    # --- SUBTYPE MODEL (Ref: Focal) ---
    df_type = df[df['canonical_subtype'].isin(['Focal', 'General'])].dropna(subset=['canonical_subtype', target_col]).copy()
    if len(df_type) > 0:
        model_type = smf.logit(f"{target_col} ~ C(canonical_subtype, Treatment('Focal'))", data=df_type).fit(disp=0)
        params, conf, pvals = model_type.params, model_type.conf_int(), model_type.pvalues
        
        results.append({
            'category': 'Epilepsy Type', 'subgroup': 'Focal', 'n': len(df_type[df_type['canonical_subtype'] == 'Focal']),
            'or': 1.0, 'low': np.nan, 'high': np.nan, 'pval': np.nan, 'is_ref': True
        })
        var_name = "C(canonical_subtype, Treatment('Focal'))[T.General]"
        if var_name in params:
            results.append({
                'category': '', 'subgroup': 'General', 'n': len(df_type[df_type['canonical_subtype'] == 'General']),
                'or': np.exp(params[var_name]), 
                'low': np.exp(conf.loc[var_name, 0]), 'high': np.exp(conf.loc[var_name, 1]), 
                'pval': pvals[var_name], 'is_ref': False
            })

    # --- CALCULATE ADJUSTED P-VALUES ---
    raw_pvals = [r['pval'] for r in results if not r['is_ref'] and not pd.isna(r['pval'])]
    if raw_pvals:
        _, adj_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')
        
        adj_idx = 0
        for r in results:
            if not r['is_ref'] and not pd.isna(r['pval']):
                r['adj_pval'] = adj_pvals[adj_idx]
                adj_idx += 1
            else:
                r['adj_pval'] = np.nan

    return results

# ==========================================================
# 4. SPREAD-OUT, 3-PANEL FOREST PLOT GENERATOR
# ==========================================================

def draw_or_forest_plot_expanded(data, title, save_path):
    if not data: return
    
    fig = plt.figure(figsize=(16, len(data) * 0.9 + 2.5))
    
    # Left Text (1.6) | Center Plot (1.0) | Right Text (0.8)
    gs = GridSpec(1, 3, width_ratios=[1.6, 1.0, 0.8], wspace=0.0) 
    ax_left = fig.add_subplot(gs[0]); ax_left.axis('off')
    ax_mid = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2]); ax_right.axis('off')
    
    y_pos = np.arange(len(data))[::-1]
    header_y = y_pos[0] + 1.2
    
    y_limits = (y_pos[-1] - 1.2, header_y + 0.6)
    ax_left.set_ylim(y_limits)
    ax_mid.set_ylim(y_limits)
    ax_right.set_ylim(y_limits)
    
    # --- 1. DRAW HEADERS ---
    ax_left.text(0.0, header_y, 'Subgroup', fontweight='bold', va='bottom', ha='left', fontsize=11)
    ax_left.text(0.5, header_y, 'No. of\nPatients', fontweight='bold', va='bottom', ha='center', fontsize=11)
    ax_left.text(0.85, header_y, 'Odds Ratio\n(95% CI)', fontweight='bold', va='bottom', ha='center', fontsize=11)
    
    ax_right.text(0.2, header_y, 'P Value', fontweight='bold', va='bottom', ha='center', fontsize=11)
    ax_right.text(0.75, header_y, 'Adjusted\nP Value', fontweight='bold', va='bottom', ha='center', fontsize=11)
    
    for ax in [ax_left, ax_mid, ax_right]:
        ax.axhline(header_y - 0.2, color='black', linewidth=1.5)
        ax.axhline(y_pos[-1] - 0.8, color='black', linewidth=1.5)
    
    # --- 2. POPULATE DATA ROWS ---
    for y, row in zip(y_pos, data):
        if row['category']:
            ax_left.text(0.0, y + 0.4, row['category'], fontweight='bold', fontsize=11, va='center')
        
        if row['is_ref']:
            or_str = "1.00 (Reference)"
            p_str, ap_str = "-", "-"
        else:
            or_str = f"{row['or']:.2f} ({row['low']:.2f}-{row['high']:.2f})"
            p_str = f"{row['pval']:.3f}" if row['pval'] >= 0.001 else "<0.001"
            ap_str = f"{row['adj_pval']:.3f}" if row['adj_pval'] >= 0.001 else "<0.001"
            
        ax_left.text(0.03, y, row['subgroup'], va='center', ha='left', fontsize=11)
        ax_left.text(0.5, y, str(row['n']), va='center', ha='center', fontsize=11)
        ax_left.text(0.85, y, or_str, va='center', ha='center', fontsize=11)
        
        ax_right.text(0.2, y, p_str, va='center', ha='center', fontsize=11)
        ax_right.text(0.75, y, ap_str, va='center', ha='center', fontsize=11)

    # --- 3. DRAW PLOT (Middle Panel) ---
    ax_mid.axvline(1.0, color='black', linewidth=1.0, zorder=1) 
    
    for y, row in zip(y_pos, data):
        if row['is_ref']:
            ax_mid.plot(1.0, y, marker='D', color='black', markersize=7, zorder=3)
        else:
            ax_mid.plot([row['low'], row['high']], [y, y], color='black', linewidth=2.0, zorder=2)
            ax_mid.plot(row['or'], y, marker='s', color='black', markersize=9, zorder=3)
            
    # --- LOG SCALE ADJUSTMENT ---
    ax_mid.set_xscale('log')
    ax_mid.set_xticks([0.25, 0.5, 1, 2, 4])
    ax_mid.set_xticklabels(['0.25', '0.50', '1', '2', '4']) # Explicit exact labels
    
    ax_mid.spines['top'].set_visible(False)
    ax_mid.spines['right'].set_visible(False)
    ax_mid.spines['left'].set_visible(False)
    ax_mid.set_yticks([])
    
    ax_mid.set_xlabel("Odds Ratio (Log Scale)", labelpad=15, fontsize=12, fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved Spread-Out Forest Plot: {save_path}")

# ==========================================================
# 5. MAIN PIPELINE
# ==========================================================

def main():
    data_dir = '/Users/edwardyao/Documents/PURM/data/'
    clinical_csv = os.path.join(data_dir, 'clinical_data_deidentified.csv')
    spike_csv = os.path.join(data_dir, 'spike_counts.csv')

    clinical_df = pd.read_csv(clinical_csv)
    spike_df = pd.read_csv(spike_csv)

    clinical_df = clinical_df.rename(columns={'patient_id': 'Patient', 'session_number': 'Session'})
    df = pd.merge(spike_df, clinical_df, on=["Patient", "Session"], how="outer")
    print(f"--- Initial Merge ---\nMerged dataset starts with {len(df)} total sessions\n")

    allowable_visits = {"CONSULT VISIT", "ESTABLISHED PATIENT VISIT", "FOLLOW-UP PATIENT CLINIC", "NEW PATIENT CLINIC", "NEW PATIENT VISIT", "NPV MANAGEMENT DURING COVID-19", "NPV NEUROLOGY", "RETURN ANNUAL VISIT", "RETURN PATIENT EXTENDED", "RETURN PATIENT VISIT", "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN"}
    
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
    df = df[(df["Duration_sec"] > 0) & (df["Duration_sec"] <= 4 * 3600)].copy()
    track_filter("Filter: Duration Bounds", before, len(df))

    before = len(df)
    df = df[df["count_0_46"].notna() & df["nlp_gender"].isin(["M", "F"])].copy()
    track_filter("Filter: Valid Spikes & Gender", before, len(df))

    bad_types = {"Uncertain if Epilepsy", "Unknown or MRN not found", "", "Non-Epileptic Seizure Disorder", "Unclassified or Unspecified"}
    before = len(df)
    df = df[~df["epilepsy_type"].isin(bad_types)].copy()
    track_filter("Filter: Valid Epilepsy Type", before, len(df))

    before = len(df)
    df["sz_freq_numeric"] = df["sz_freqs"].apply(parse_sz_freq)
    df = df[df["sz_freq_numeric"].notna()].copy()
    track_filter("Filter: Valid Seizure Frequency", before, len(df))

    df["canonical_subtype"] = df.apply(canonical_subtype, axis=1)

    patient_df = df.groupby(["Patient", "nlp_gender", "epilepsy_type", "canonical_subtype"], dropna=False).agg(
        total_spikes=("count_0_46", "sum"),
        total_duration=("Duration_sec", "sum"),
        mean_sz_freq=("sz_freq_numeric", "mean") 
    ).reset_index()

    patient_df["spike_rate_per_hour"] = (patient_df["total_spikes"] / patient_df["total_duration"]) * 3600
    
    before_drop = len(patient_df)
    patient_df = patient_df.dropna(subset=["spike_rate_per_hour", "mean_sz_freq"])
    print("--- Aggregating to Patient Level ---")
    print(f"FINAL PATIENT COHORT SIZE: {len(patient_df)}\n")

    # STEP A: CONTINUOUS STATS (Print Output)
    cohorts = {
        "Overall Cohort": patient_df,
        "General Epilepsy": patient_df[patient_df["epilepsy_type"] == "General"],
        "Focal Epilepsy": patient_df[patient_df["epilepsy_type"] == "Focal"]
    }

    metrics = {
        "spike_rate_per_hour": "Spike Rate (per hour)",
        "mean_sz_freq": "Average Seizure Frequency"
    }

    for metric_col, metric_name in metrics.items():
        for cohort_name, cohort_data in cohorts.items():
            run_cohort_analysis(cohort_data, metric_col, f"{cohort_name} | {metric_name}")


    # STEP B: WIDE OR FOREST PLOTS WITH EXPLICIT THRESHOLDS
    print("\n======================================================")
    print("=== GENERATING WIDE OR FOREST PLOTS ===")
    print("======================================================")
    
    # --- UPDATED: Explicit Hardcoded Clinical Thresholds ---
    clinical_spike_threshold = 1.79
    clinical_sz_threshold = 0.73
    
    patient_df["high_spike_rate"] = (patient_df["spike_rate_per_hour"] > clinical_spike_threshold).astype(int)
    patient_df["high_sz_freq"] = (patient_df["mean_sz_freq"] > clinical_sz_threshold).astype(int)

    spike_data = run_logistic_models(patient_df, "high_spike_rate")
    draw_or_forest_plot_expanded(
        spike_data, 
        f"Odds of High Spike Rate (> {clinical_spike_threshold} spikes/hr)", 
        os.path.join(data_dir, "ForestPlot_OR_SpikeRate_Wide.png")
    )
    
    sz_data = run_logistic_models(patient_df, "high_sz_freq")
    draw_or_forest_plot_expanded(
        sz_data, 
        f"Odds of High Seizure Frequency (> {clinical_sz_threshold} seizures/month)", 
        os.path.join(data_dir, "ForestPlot_OR_SzFreq_Wide.png")
    )

if __name__ == "__main__":
    main()