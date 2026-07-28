import pandas as pd
import numpy as np
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
# 2. QUANTILE REGRESSION STATS (MEDIANS) WITH BOOTSTRAPPING
# ==========================================================

def get_quantile_bootstrapped_stats(df, target_col, n_bootstraps=2000):
    results = []
    
    # --- SEX MODEL (Ref: Female) ---
    df_sex = df.dropna(subset=['nlp_gender', target_col]).copy()
    if len(df_sex) > 0:
        # Female Reference Baseline Median
        results.append({
            'category': 'Sex', 'subgroup': 'Female', 'n': len(df_sex[df_sex['nlp_gender'] == 'F']),
            'median': df_sex[df_sex['nlp_gender'] == 'F'][target_col].median(), 
            'diff': 0.0, 'low': np.nan, 'high': np.nan, 'pval': np.nan, 'is_ref': True
        })
        
        # Quantile Regression for the Median (q=0.5)
        model_sex = smf.quantreg(f"{target_col} ~ C(nlp_gender, Treatment('F'))", data=df_sex).fit(q=0.5)
        var_name = "C(nlp_gender, Treatment('F'))[T.M]"
        
        if var_name in model_sex.params:
            point_est = model_sex.params[var_name]
            p_val = model_sex.pvalues[var_name]
            
            # Bootstrap the Median Regression Coefficient
            np.random.seed(42)
            boot_coefs = []
            for _ in range(n_bootstraps):
                sample_df = df_sex.sample(frac=1, replace=True)
                try:
                    b_model = smf.quantreg(f"{target_col} ~ C(nlp_gender, Treatment('F'))", data=sample_df).fit(q=0.5)
                    boot_coefs.append(b_model.params[var_name])
                except:
                    pass
            
            results.append({
                'category': '', 'subgroup': 'Male', 'n': len(df_sex[df_sex['nlp_gender'] == 'M']),
                'median': df_sex[df_sex['nlp_gender'] == 'M'][target_col].median(), 
                'diff': point_est, 'low': np.percentile(boot_coefs, 2.5), 'high': np.percentile(boot_coefs, 97.5), 
                'pval': p_val, 'is_ref': False
            })

    # --- SUBTYPE MODEL (Ref: Focal) ---
    df_type = df[df['canonical_subtype'].isin(['Focal', 'General'])].dropna(subset=['canonical_subtype', target_col]).copy()
    if len(df_type) > 0:
        # Focal Reference Baseline Median
        results.append({
            'category': 'Epilepsy Type', 'subgroup': 'Focal', 'n': len(df_type[df_type['canonical_subtype'] == 'Focal']),
            'median': df_type[df_type['canonical_subtype'] == 'Focal'][target_col].median(), 
            'diff': 0.0, 'low': np.nan, 'high': np.nan, 'pval': np.nan, 'is_ref': True
        })
        
        # Quantile Regression for the Median (q=0.5)
        model_type = smf.quantreg(f"{target_col} ~ C(canonical_subtype, Treatment('Focal'))", data=df_type).fit(q=0.5)
        var_name = "C(canonical_subtype, Treatment('Focal'))[T.General]"
        
        if var_name in model_type.params:
            point_est = model_type.params[var_name]
            p_val = model_type.pvalues[var_name]
            
            # Bootstrap the Median Regression Coefficient
            np.random.seed(42)
            boot_coefs = []
            for _ in range(n_bootstraps):
                sample_df = df_type.sample(frac=1, replace=True)
                try:
                    b_model = smf.quantreg(f"{target_col} ~ C(canonical_subtype, Treatment('Focal'))", data=sample_df).fit(q=0.5)
                    boot_coefs.append(b_model.params[var_name])
                except:
                    pass
                     
            results.append({
                'category': '', 'subgroup': 'General', 'n': len(df_type[df_type['canonical_subtype'] == 'General']),
                'median': df_type[df_type['canonical_subtype'] == 'General'][target_col].median(), 
                'diff': point_est, 'low': np.percentile(boot_coefs, 2.5), 'high': np.percentile(boot_coefs, 97.5), 
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
# 3. CONTINUOUS FOREST PLOT GENERATOR
# ==========================================================

def draw_quantreg_forest_plot(data, title, save_path, xmin=None, xmax=None, xticks=None):
    if not data: 
        return
    
    fig = plt.figure(figsize=(16, len(data) * 0.9 + 2.5))
    
    gs = GridSpec(1, 3, width_ratios=[1.6, 1.0, 0.8], wspace=0.0) 
    ax_left = fig.add_subplot(gs[0]); ax_left.axis('off')
    ax_mid = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2]); ax_right.axis('off')
    
    y_pos = np.arange(len(data))
    header_y = -1.0
    
    y_limits = (len(data) - 0.2, -1.5)
    for ax in [ax_left, ax_mid, ax_right]: 
        ax.set_ylim(y_limits)
    
    # --- 1. DRAW HEADERS ---
    ax_left.text(0.0, header_y, 'Subgroup', fontweight='bold', va='bottom', ha='left', fontsize=11)
    ax_left.text(0.5, header_y, 'No. of\nPatients', fontweight='bold', va='bottom', ha='center', fontsize=11)
    ax_left.text(0.85, header_y, 'Median Diff.\n(95% CI)', fontweight='bold', va='bottom', ha='center', fontsize=11)
    
    ax_right.text(0.2, header_y, 'P Value', fontweight='bold', va='bottom', ha='center', fontsize=11)
    ax_right.text(0.75, header_y, 'Adjusted\nP Value', fontweight='bold', va='bottom', ha='center', fontsize=11)
    
    for ax in [ax_left, ax_mid, ax_right]:
        ax.axhline(header_y - 0.1, color='black', linewidth=1.5)
        ax.axhline(len(data) - 0.5, color='black', linewidth=1.5)
    
    # --- 2. POPULATE DATA ROWS ---
    for y, row in zip(y_pos, data):
        if row['category']:
            ax_left.text(0.0, y - 0.4, row['category'], fontweight='bold', fontsize=11, va='center')
        
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

    # --- 3. DRAW PLOT ---
    ax_mid.axvline(0.0, color='black', linewidth=1.0, linestyle='--', zorder=1) 
    
    for y, row in zip(y_pos, data):
        if row['is_ref']:
            ax_mid.plot(0.0, y, marker='D', color='black', markersize=7, zorder=3)
        else:
            ax_mid.plot([row['low'], row['high']], [y, y], color='black', linewidth=2.0, zorder=2)
            ax_mid.plot(row['diff'], y, marker='s', color='black', markersize=9, zorder=3)
            
    if xmin is not None and xmax is not None:
        ax_mid.set_xlim(xmin, xmax)
    if xticks is not None:
        ax_mid.set_xticks(xticks)
    
    ax_mid.spines['top'].set_visible(False)
    ax_mid.spines['right'].set_visible(False)
    ax_mid.spines['left'].set_visible(False)
    ax_mid.set_yticks([])
    
    ax_mid.set_xlabel("Difference in Medians (Quantile Regression Coef)\n<-- Favors Reference Group   |   Favors Comparison Group -->", labelpad=15, fontsize=11, fontweight='bold')
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved Quantile Regression Forest Plot: {save_path}")

# ==========================================================
# 4. MAIN PIPELINE
# ==========================================================

def main():
    data_dir = '/Users/edwardyao/Documents/PURM/data/'
    clinical_csv = os.path.join(data_dir, 'clinical_data_deidentified.csv')
    spike_csv = os.path.join(data_dir, 'spike_counts.csv')

    clinical_df = pd.read_csv(clinical_csv)
    spike_df = pd.read_csv(spike_csv)

    clinical_df = clinical_df.rename(columns={'patient_id': 'Patient', 'session_number': 'Session'})
    df = pd.merge(spike_df, clinical_df, on=["Patient", "Session"], how="outer")

    allowable_visits = {"CONSULT VISIT", "ESTABLISHED PATIENT VISIT", "FOLLOW-UP PATIENT CLINIC", "NEW PATIENT CLINIC", "NEW PATIENT VISIT", "NPV MANAGEMENT DURING COVID-19", "NPV NEUROLOGY", "RETURN ANNUAL VISIT", "RETURN PATIENT EXTENDED", "RETURN PATIENT VISIT", "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN"}
    df = df[df["visit_type"].astype(str).apply(lambda v: any(x in v for x in allowable_visits))].copy()
    
    acq = df["acquired_on"].fillna("").astype(str).str.lower()
    patient_class = df["report_PATIENT_CLASS"].fillna("").astype(str).str.lower()
    jay = df["jay_in_or_out"].fillna("").astype(str).str.lower()
    df = df[acq.str.contains("spe") | acq.str.contains("radnor") | (patient_class == "outpatient") | (jay == "out")].copy()

    df = df[(df["Duration_sec"] > 0) & (df["Duration_sec"] <= 4 * 3600)].copy()
    df = df[df["count_0_46"].notna() & df["nlp_gender"].isin(["M", "F"])].copy()

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

    # 1. Spike Rate Plot (Configured from -1.0 to 2.5)
    spike_data = get_quantile_bootstrapped_stats(patient_df, "spike_rate_per_hour")
    draw_quantreg_forest_plot(
        data=spike_data, 
        title="Differences in Median Spike Rate (Quantile Regression)", 
        save_path=os.path.join(data_dir, "ForestPlot_QuantReg_SpikeRate.png"),
        xmin=-1.0, 
        xmax=2.5, 
        xticks=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    )
    
    # 2. Seizure Frequency Plot (Configured from -2.0 to 5.0)
    sz_data = get_quantile_bootstrapped_stats(patient_df, "mean_sz_freq")
    draw_quantreg_forest_plot(
        data=sz_data, 
        title="Differences in Median Seizure Frequency (Quantile Regression)", 
        save_path=os.path.join(data_dir, "ForestPlot_QuantReg_SzFreq.png"),
        xmin=-2.0, 
        xmax=5.0, 
        xticks=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    )

if __name__ == "__main__":
    main()