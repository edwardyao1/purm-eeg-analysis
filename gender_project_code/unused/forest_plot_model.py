import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
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
    if pd.isna(val): 
        return np.nan
    
    val_str = str(val).strip().strip("[]").replace("'", "").replace('"', "")
    
    if not val_str: 
        return np.nan
    
    try:
        parts = [float(x.strip()) for x in val_str.split(",") if x.strip() and x.strip().lower() != 'null']
        return np.mean(parts) if parts else np.nan
    except ValueError:
        return np.nan
    
def canonical_subtype(row):
    etype = str(row.get("epilepsy_type", "")).lower()
    
    if "focal" in etype: 
        return "Focal"
    if etype == "general": 
        return "General"
    
    return np.nan

# ==========================================================
# 2. CONTINUOUS STATS: BOOTSTRAP MEDIAN DIFFERENCE
# ==========================================================

def get_continuous_stats(df, target_col):
    results = []
    
    # --- SEX MODEL (Ref: Female) ---
    df_sex = df.dropna(subset=['nlp_gender', target_col]).copy()
    
    # Ensures there are patients in the sample
    if len(df_sex) > 0:
        females = df_sex[df_sex['nlp_gender'] == 'F'][target_col].values
        males = df_sex[df_sex['nlp_gender'] == 'M'][target_col].values
        
        # Female Reference
        results.append({
            'category': 'Sex', 'subgroup': 'Female', 'n': len(females),
            'median': np.median(females), 'diff': 0.0, 'low': np.nan, 'high': np.nan, 
            'pval': np.nan, 'is_ref': True
        })
        
        # Male Comparison (Male - Female)
        if len(males) > 0 and len(females) > 0:
            stat, p_val = mannwhitneyu(males, females, alternative='two-sided')
            
            # Bootstrap the difference by 10000
            np.random.seed(42)
            diffs = [np.median(np.random.choice(males, len(males), replace=True)) - 
                     np.median(np.random.choice(females, len(females), replace=True)) for _ in range(10000)]
            
            results.append({
                'category': '', 'subgroup': 'Male', 'n': len(males),
                'median': np.median(males), 'diff': np.median(males) - np.median(females), 
                'low': np.percentile(diffs, 2.5), 'high': np.percentile(diffs, 97.5), 
                'pval': p_val, 'is_ref': False
            })

    # --- SUBTYPE MODEL (Ref: Focal) ---
    df_type = df[df['canonical_subtype'].isin(['Focal', 'General'])].dropna(subset=['canonical_subtype', target_col]).copy()
    
    if len(df_type) > 0:
        focal = df_type[df_type['canonical_subtype'] == 'Focal'][target_col].values
        general = df_type[df_type['canonical_subtype'] == 'General'][target_col].values
        
        # Focal Reference
        results.append({
            'category': 'Epilepsy Type', 'subgroup': 'Focal', 'n': len(focal),
            'median': np.median(focal), 'diff': 0.0, 'low': np.nan, 'high': np.nan, 
            'pval': np.nan, 'is_ref': True
        })
        
        # General Comparison (General - Focal)
        if len(general) > 0 and len(focal) > 0:
            stat, p_val = mannwhitneyu(general, focal, alternative='two-sided')
            
            # Bootstrapping the difference by 10000
            np.random.seed(42)
            diffs = [np.median(np.random.choice(general, len(general), replace=True)) - 
                     np.median(np.random.choice(focal, len(focal), replace=True)) for _ in range(10000)]
                     
            results.append({
                'category': '', 'subgroup': 'General', 'n': len(general),
                'median': np.median(general), 'diff': np.median(general) - np.median(focal), 
                'low': np.percentile(diffs, 2.5), 'high': np.percentile(diffs, 97.5), 
                'pval': p_val, 'is_ref': False
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
# 3. CONTINUOUS FOREST PLOT GENERATOR (With Dynamic Scale)
# ==========================================================

def draw_continuous_forest_plot(data, title, save_path, xmin=None, xmax=None, xticks=None):
    if not data: 
        return
    
    fig = plt.figure(figsize=(16, len(data) * 0.9 + 2.5))
    
    # Left Text (1.6) | Center Plot (1.0) | Right Text (0.8)
    gs = GridSpec(1, 3, width_ratios=[1.6, 1.0, 0.8], wspace=0.0) 
    ax_left = fig.add_subplot(gs[0]); ax_left.axis('off')
    ax_mid = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2]); ax_right.axis('off')
    
    y_pos = np.arange(len(data))[::-1]
    header_y = y_pos[0] + 1.2
    
    y_limits = (y_pos[-1] - 1.2, header_y + 0.6)
    for ax in [ax_left, ax_mid, ax_right]: ax.set_ylim(y_limits)
    
    # --- 1. DRAW HEADERS ---
    ax_left.text(0.0, header_y, 'Subgroup', fontweight='bold', va='bottom', ha='left', fontsize=11)
    ax_left.text(0.5, header_y, 'No. of\nPatients', fontweight='bold', va='bottom', ha='center', fontsize=11)
    ax_left.text(0.85, header_y, 'Median Diff.\n(95% CI)', fontweight='bold', va='bottom', ha='center', fontsize=11)
    
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
            diff_str = f"Reference (Med: {row['median']:.1f})"
            p_str, ap_str = "-", "-"
        else:
            diff_str = f"{row['diff']:.2f} ({row['low']:.2f} to {row['high']:.2f})"
            p_str = f"{row['pval']:.3f}" if row['pval'] >= 0.001 else "<0.001"
            ap_str = f"{row['adj_pval']:.3f}" if row['adj_pval'] >= 0.001 else "<0.001"
            
        ax_left.text(0.03, y, row['subgroup'], va='center', ha='left', fontsize=11)
        ax_left.text(0.5, y, str(row['n']), va='center', ha='center', fontsize=11)
        ax_left.text(0.85, y, diff_str, va='center', ha='center', fontsize=11)
        
        ax_right.text(0.2, y, p_str, va='center', ha='center', fontsize=11)
        ax_right.text(0.75, y, ap_str, va='center', ha='center', fontsize=11)

    # --- 3. DRAW PLOT (Middle Panel - LINEAR SCALE) ---
    ax_mid.axvline(0.0, color='black', linewidth=1.0, linestyle='--', zorder=1) # Ref Line at 0
    
    for y, row in zip(y_pos, data):
        if row['is_ref']:
            ax_mid.plot(0.0, y, marker='D', color='black', markersize=7, zorder=3)
        else:
            ax_mid.plot([row['low'], row['high']], [y, y], color='black', linewidth=2.0, zorder=2)
            ax_mid.plot(row['diff'], y, marker='s', color='black', markersize=9, zorder=3)
            
    # --- Dynamic Scaling Adjustments ---
    if xmin is not None and xmax is not None:
        ax_mid.set_xlim(xmin, xmax)
    if xticks is not None:
        ax_mid.set_xticks(xticks)
    
    ax_mid.spines['top'].set_visible(False)
    ax_mid.spines['right'].set_visible(False)
    ax_mid.spines['left'].set_visible(False)
    ax_mid.set_yticks([])
    
    ax_mid.set_xlabel("Difference in Medians (Linear Scale)\n<-- Favors Reference Group   |   Favors Comparison Group -->", labelpad=15, fontsize=11, fontweight='bold')
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved Continuous Forest Plot: {save_path}")

# ==========================================================
# 4. MAIN PIPELINE
# ==========================================================

def main():
    data_dir = '/Users/edwardyao/Documents/PURM/data/'
    save_dir = '/Users/edwardyao/Documents/PURM/gender_project_output/'
    clinical_csv = os.path.join(data_dir, 'clinical_data_deidentified.csv')
    spike_csv = os.path.join(data_dir, 'spike_counts.csv')

    clinical_df = pd.read_csv(clinical_csv)
    spike_df = pd.read_csv(spike_csv)

    clinical_df = clinical_df.rename(columns={'patient_id': 'Patient', 'session_number': 'Session'})
    df = pd.merge(spike_df, clinical_df, on=["Patient", "Session"], how="outer")
    print(f"--- Initial Merge ---\nMerged dataset starts with {len(df)} total sessions\n")

    allowable_visits = {"CONSULT VISIT", "ESTABLISHED PATIENT VISIT", "FOLLOW-UP PATIENT CLINIC", "NEW PATIENT CLINIC", "NEW PATIENT VISIT", "NPV MANAGEMENT DURING COVID-19", "NPV NEUROLOGY", "RETURN ANNUAL VISIT", "RETURN PATIENT EXTENDED", "RETURN PATIENT VISIT", "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN"}
    
    df = df[df["visit_type"].astype(str).apply(lambda v: any(x in v for x in allowable_visits))].copy()
    
    acq = df["acquired_on"].fillna("").astype(str).str.lower()
    patient_class = df["report_PATIENT_CLASS"].fillna("").astype(str).str.lower()
    jay = df["jay_in_or_out"].fillna("").astype(str).str.lower()
    df = df[acq.str.contains("spe") | acq.str.contains("radnor") | (patient_class == "outpatient") | (jay == "out")].copy()

    # Filter that the EEG length is greater than 0 and less than 4 hours.
    df = df[(df["Duration_sec"] > 0) & (df["Duration_sec"] <= 4 * 3600)].copy()
    
    # Filter that it has a spike rate and male or female sex.
    df = df[df["count_0_46"].notna() & df["nlp_gender"].isin(["M", "F"])].copy()

    # Filter that it does not contain any uncertain or non-epileptic classifications.
    bad_types = {"Uncertain if Epilepsy", "Unknown or MRN not found", "", "Non-Epileptic Seizure Disorder", "Unclassified or Unspecified"}
    df = df[~df["epilepsy_type"].isin(bad_types)].copy()
    
    df["sz_freq_numeric"] = df["sz_freqs"].apply(parse_sz_freq)
    df = df[df["sz_freq_numeric"].notna()].copy()
    df["canonical_subtype"] = df.apply(canonical_subtype, axis=1)

    patient_df = df.groupby(["Patient", "nlp_gender", "epilepsy_type", "canonical_subtype"], dropna=False).agg(
        total_spikes=("count_0_46", "sum"),
        total_duration=("Duration_sec", "sum"),
        mean_sz_freq=("sz_freq_numeric", "mean") 
    ).reset_index()
    
    patient_df["spike_rate_per_hour"] = (patient_df["total_spikes"] / patient_df["total_duration"]) * 3600
    patient_df = patient_df.dropna(subset=["spike_rate_per_hour", "mean_sz_freq"])
    print(f"FINAL PATIENT COHORT SIZE: {len(patient_df)}\n")

    print("\n======================================================")
    print("=== GENERATING CONTINUOUS FOREST PLOTS ===")
    print("======================================================")

    # 1. Spike Rate Plot (Configured from -1.0 to 2.5)
    spike_data = get_continuous_stats(patient_df, "spike_rate_per_hour")
    draw_continuous_forest_plot(
        data=spike_data, 
        title="Differences in Median Spike Rate (per hour)", 
        save_path=os.path.join(save_dir, "ForestPlot_Continuous_SpikeRate.png"),
        xmin=-1.0, 
        xmax=2.5, 
        xticks=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    )
    
    # # 2. Seizure Frequency Plot (Configured from -2.0 to 5.0)
    # sz_data = get_continuous_stats(patient_df, "mean_sz_freq")
    # draw_continuous_forest_plot(
    #     data=sz_data, 
    #     title="Differences in Median Seizure Frequency", 
    #     save_path=os.path.join(data_dir, "ForestPlot_Continuous_SzFreq.png"),
    #     xmin=-2.0, 
    #     xmax=5.0, 
    #     xticks=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # )

if __name__ == "__main__":
    main()