# Overall men vs. women spike rate and seizure frequency
# Creates scatter plots with Mann-Whitney U test results for overall and stratified comparisons
# Includes comprehensive descriptive statistics (Medians, Quartiles, IQRs, Min/Max)
# Effect sizes for Mann-Whitney U are calculated as Rank-Biserial Correlation (r_rb)

import pandas as pd
import numpy as np
import os
import ast
import warnings
from scipy.stats import mannwhitneyu, gaussian_kde, chi2_contingency
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

# ==========================================================
# TRACKING & PARSING HELPERS
# ==========================================================
def track_patients(step_name, before_pats, after_pats):
    dropped = len(before_pats) - len(after_pats)
    print(f"--- {step_name} ---")
    print(f"  Patients Before: {len(before_pats)}")
    print(f"  Patients After:  {len(after_pats)}")
    print(f"  Total Dropped:   {dropped}\n")

def parse_json_array(val, is_numeric=False, is_hassz=False):
    if pd.isna(val) or val == "": return []
    val_str = str(val).strip()
    if val_str in ("[]", "[null]", "<missing>"): return []
    
    val_str = val_str.replace('null', 'None')
    try:
        parsed = ast.literal_eval(val_str)
        if isinstance(parsed, list):
            if is_numeric:
                res = []
                for x in parsed:
                    if x is None:
                        res.append(np.nan)
                    else:
                        fv = float(x)
                        if is_hassz and fv == 2.0:
                            res.append(np.nan)
                        elif not is_hassz and fv < 0:
                            res.append(np.nan)
                        else:
                            res.append(fv)
                return res
            return [str(x).strip() for x in parsed]
        return []
    except Exception:
        return []

def assign_canonical_subtype(row):
    etype = str(row.get("epilepsy_type", "")).lower()
    espec = str(row.get("epilepsy_specific", "")).lower()
    
    if "focal" in etype or "temporal" in espec or "frontal" in espec:
        return "Focal"
    if etype == "general" or etype == "generalized":
        return "Generalized"
    return np.nan

# ==========================================================
# STATISTICS & PLOTTING HELPERS
# ==========================================================
def print_descriptive_stats(df, target_var, target_name):
    """Prints a structured table of medians, quartiles, IQRs, min/max, and mean/SD."""
    print(f"\n=========================================================================================================================")
    print(f"DESCRIPTIVE STATISTICS: {target_name.upper()}")
    print(f"=========================================================================================================================")
    header = f"{'Cohort / Group':<20} | {'N':>5} | {'Median':>10} | {'Q1 (25%)':>10} | {'Q3 (75%)':>10} | {'IQR':>10} | {'Min':>10} | {'Max':>10} | {'Mean ± SD':>18}"
    print(header)
    print("-" * len(header))
    
    def print_row(sub_df, label):
        s = sub_df[target_var].dropna()
        if len(s) == 0:
            print(f"{label:<20} | {0:>5} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>18}")
            return
        med = s.median()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        mn = s.min()
        mx = s.max()
        mean = s.mean()
        std = s.std()
        mean_sd = f"{mean:.2f} ± {std:.2f}"
        print(f"{label:<20} | {len(s):>5d} | {med:>10.4f} | {q1:>10.4f} | {q3:>10.4f} | {iqr:>10.4f} | {mn:>10.4f} | {mx:>10.4f} | {mean_sd:>18}")

    print_row(df, "Overall - All")
    print_row(df[df['nlp_gender'] == 'F'], "Overall - Female")
    print_row(df[df['nlp_gender'] == 'M'], "Overall - Male")
    print("-" * len(header))
    
    for subtype in ['Focal', 'Generalized']:
        sub_df = df[df['canonical_subtype'] == subtype]
        print_row(sub_df, f"{subtype} - All")
        print_row(sub_df[sub_df['nlp_gender'] == 'F'], f"{subtype} - Female")
        print_row(sub_df[sub_df['nlp_gender'] == 'M'], f"{subtype} - Male")
        if subtype != 'Generalized':
            print("-" * len(header))
    print("=========================================================================================================================\n")

def mwu_rank_biserial(df, group_col, value_col, g1="M", g2="F"):
    """
    Computes Mann-Whitney U and its corresponding Rank-Biserial Correlation.
    r_rb ranges from -1 to 1. Positive values indicate g1 > g2.
    """
    x = df[df[group_col] == g1][value_col].dropna()
    y = df[df[group_col] == g2][value_col].dropna()
    n1, n2 = len(x), len(y)
    
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, n1, n2

    u, p = mannwhitneyu(x, y, alternative="two-sided")
    # Rank-biserial correlation formula from U statistic
    r_rb = np.nan if n1 * n2 == 0 else (2 * u) / (n1 * n2) - 1
    return u, p, r_rb, n1, n2

def run_chi_square_analysis(df):
    print("\n=== CHI-SQUARE TEST: SEX VS. EPILEPSY TYPE ===")
    
    obs_table = pd.crosstab(
        df['nlp_gender'], 
        df['canonical_subtype']
    )
    
    table_with_margins = pd.crosstab(
        df['nlp_gender'], 
        df['canonical_subtype'], 
        margins=True, 
        margins_name='Total'
    )
    print("Contingency Table (Observed Counts):")
    print(table_with_margins)
    print()
    
    chi2, p, dof, expected = chi2_contingency(obs_table, correction=True)
    
    n = obs_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(obs_table.shape) - 1)))
    
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"p-value:              {p:.6f}")
    print(f"Degrees of Freedom:   {dof}")
    print(f"Effect Size (Cramér's V): {cramers_v:.4f}")
    
    expected_df = pd.DataFrame(expected, index=obs_table.index, columns=obs_table.columns)
    print("\nExpected Counts (under null hypothesis of independence):")
    print(expected_df.round(2))
    
    if (expected < 5).any():
        from scipy.stats import fisher_exact
        res = fisher_exact(obs_table)
        print("\nWARNING: One or more expected cell counts are < 5.")
        print(f"Fisher's Exact Test p-value: {res.pvalue:.6f} (Odds Ratio: {res.statistic:.4f})")

def calculate_density_jitter(y_vals, width=0.3):
    if len(y_vals) < 3: return np.zeros(len(y_vals))
    y_log = np.log1p(y_vals)
    y_log_noisy = y_log + np.random.normal(0, 1e-6, size=len(y_log))
    
    try:
        kde = gaussian_kde(y_log_noisy)
        density = kde(y_log_noisy)
        if density.max() > 0:
            density = density / density.max()
        return np.random.uniform(-width, width, size=len(y_vals)) * density
    except Exception:
        return np.random.uniform(-width, width, size=len(y_vals))

def plot_scatter_with_stats(df, target_var, title, ylabel, save_path):
    u, p, r_rb, n1, n2 = mwu_rank_biserial(df, "nlp_gender", target_var, g1="M", g2="F")
    if n1 == 0 or n2 == 0:
        print(f"Skipping plot for {title} due to insufficient data.")
        return

    plot_df = df.copy()
    plot_df['Sex'] = plot_df['nlp_gender'].map({'F': 'Female', 'M': 'Male'})
    plot_df[target_var] = plot_df[target_var].clip(lower=0) 
    
    plt.figure(figsize=(8, 8))
    sns.set_theme(style="ticks") 
    
    ax = sns.boxplot(
        x="Sex", y=target_var, data=plot_df, order=["Female", "Male"], showfliers=False, width=0.4, 
        boxprops=dict(facecolor="lightgray", edgecolor="black", linewidth=2.5, alpha=0.7),
        medianprops=dict(color="black", linewidth=3.5),
        whiskerprops=dict(color="black", linewidth=2.5),
        capprops=dict(color="black", linewidth=2.5)
    )

    custom_palette = {"Female": "#e24a33", "Male": "#348abd"}
    for i, group in enumerate(["Female", "Male"]):
        group_mask = plot_df['Sex'] == group
        y_vals = plot_df.loc[group_mask, target_var].values
        if len(y_vals) > 0:
            jitter = calculate_density_jitter(y_vals, width=0.25)
            ax.scatter(i + jitter, y_vals, color=custom_palette[group], s=20, alpha=0.6, edgecolors='none', zorder=3)
    
    plt.yscale('symlog', linthresh=0.01)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.7)
    sns.despine()

    # Font size augmentation
    plt.title(title, fontsize=18, fontweight='bold', pad=15)
    plt.ylabel(f"{ylabel} (Log Scale)", fontsize=16, fontweight='bold')
    plt.xlabel("Sex", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=14)
    
    stats_text = f"Mann-Whitney U = {u:.2f}\np-value = {p:.4f}\nRank-Biserial r = {r_rb:.4f}\nn (Female) = {n2}\nn (Male) = {n1}"
    plt.annotate(stats_text, xy=(0.95, 0.05), xycoords='axes fraction', horizontalalignment='right', 
                 verticalalignment='bottom', fontsize=13, bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="lightgray", lw=1, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_stratified_scatter_with_stats(df, target_var, title, ylabel, save_path):
    plot_df = df.copy()
    plot_df['Sex'] = plot_df['nlp_gender'].map({'F': 'Female', 'M': 'Male'})
    plot_df[target_var] = plot_df[target_var].clip(lower=0) 
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    sns.set_theme(style="ticks") 
    custom_palette = {"Female": "#e24a33", "Male": "#348abd"}
    
    for ax, subtype in zip(axes, ["Focal", "Generalized"]):
        sub_df = plot_df[plot_df['canonical_subtype'] == subtype]
        u, p, r_rb, n1, n2 = mwu_rank_biserial(sub_df, "nlp_gender", target_var, g1="M", g2="F")
        
        sns.boxplot(
            x="Sex", y=target_var, data=sub_df, order=["Female", "Male"], showfliers=False, width=0.4, 
            boxprops=dict(facecolor="lightgray", edgecolor="black", linewidth=2.5, alpha=0.7),
            medianprops=dict(color="black", linewidth=3.5),
            whiskerprops=dict(color="black", linewidth=2.5),
            capprops=dict(color="black", linewidth=2.5), ax=ax
        )

        for i, group in enumerate(["Female", "Male"]):
            group_mask = sub_df['Sex'] == group
            y_vals = sub_df.loc[group_mask, target_var].values
            if len(y_vals) > 0:
                jitter = calculate_density_jitter(y_vals, width=0.25)
                ax.scatter(i + jitter, y_vals, color=custom_palette[group], s=20, alpha=0.6, edgecolors='none', zorder=3)
        
        ax.set_yscale('symlog', linthresh=0.01)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.7)
        
        ax.set_title(f"{subtype} Epilepsy", fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel("Sex", fontsize=16, fontweight='bold')
        if ax == axes[0]: 
            ax.set_ylabel(f"{ylabel} (Log Scale)", fontsize=16, fontweight='bold')
            
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=14)
        
        if n1 > 0 and n2 > 0:
            stats_text = f"Mann-Whitney U = {u:.2f}\np-value = {p:.4f}\nRank-Biserial r = {r_rb:.4f}\nn (Female) = {n2}\nn (Male) = {n1}"
            ax.annotate(stats_text, xy=(0.95, 0.05), xycoords='axes fraction', horizontalalignment='right', 
                        verticalalignment='bottom', fontsize=13, bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="lightgray", lw=1, alpha=0.9))

    sns.despine()
    fig.suptitle(title, fontsize=20, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_statistical_analysis(patient_df, target_var, target_name):
    print(f"\n=== MANN-WHITNEY U: {target_name.upper()} ===")
    u, p, r_rb, n1, n2 = mwu_rank_biserial(patient_df, "nlp_gender", target_var)
    print(f"Overall -> M = {n1}, F = {n2} | U = {u:.2f}, p = {p:.6f}, r_rb = {r_rb:.4f}")

    type_pvals, type_results = [], []
    for t in ["Focal", "Generalized"]:
        sub = patient_df[patient_df["canonical_subtype"] == t]
        u, p, r_rb, n1, n2 = mwu_rank_biserial(sub, "nlp_gender", target_var)
        type_results.append((t, u, p, r_rb, n1, n2))
        type_pvals.append(1.0 if pd.isna(p) else p)

    reject, p_adj, _, _ = multipletests(type_pvals, method="bonferroni")
    for (t, u, p, r_rb, n1, n2), pa, sig in zip(type_results, p_adj, reject):
        print(f"Subtype: {t} -> M = {n1}, F = {n2} | U = {u:.2f}, Raw p = {p:.6f}, Bonf p = {pa:.6f}, r_rb = {r_rb:.4f}, Sig = {sig}")
        
# ==========================================================
# MAIN PIPELINE
# ==========================================================
def main():
    data_dir = '/Users/edwardyao/Documents/PURM/data/'
    save_dir = '/Users/edwardyao/Documents/PURM/gender_project_output/'
    os.makedirs(save_dir, exist_ok=True)
    
    clinical_csv = os.path.join(data_dir, 'clinical_data_deidentified.csv')
    spike_csv = os.path.join(data_dir, 'spike_counts.csv')

    print("=" * 75)
    print("LOADING DATA")
    print("=" * 75)

    clinical_df = pd.read_csv(clinical_csv, low_memory=False).rename(columns={'patient_id': 'Patient', 'session_number': 'Session'})
    spike_df = pd.read_csv(spike_csv, low_memory=False)

    print(f"  spike_counts:           {len(spike_df):6,} sessions")
    print(f"  clinical_data:          {len(clinical_df):6,} records\n")

    print("Calculating patient ages from de-identified dates...")
    clinical_df['start_time_deid'] = pd.to_datetime(clinical_df['start_time_deid'], errors='coerce')
    clinical_df['deid_birth_date'] = pd.to_datetime(clinical_df['deid_birth_date'], errors='coerce')
    clinical_df['age'] = (clinical_df['start_time_deid'] - clinical_df['deid_birth_date']).dt.days / 365.25

    print("=" * 75)
    print("DATA CLEANING & COHORT ATTRITION")
    print("=" * 75)

    current_patients = set(clinical_df['Patient'].dropna().unique())
    print(f"Starting Cohort: {len(current_patients)} unique patients\n")

    # 1. OUTPATIENT & ROUTINE FILTER
    before_pats = current_patients.copy()
    session_df = pd.merge(spike_df, clinical_df[['Patient', 'Session', 'acquired_on', 'report_PATIENT_CLASS', 'jay_in_or_out', 'age']], on=["Patient", "Session"], how="inner")
    
    acq = session_df["acquired_on"].fillna("").astype(str).str.lower()
    p_class = session_df["report_PATIENT_CLASS"].fillna("").astype(str).str.lower()
    jay = session_df["jay_in_or_out"].fillna("").astype(str).str.lower()
    
    is_outpt = (acq.str.contains("spe") | acq.str.contains("radnor") | (p_class == "outpatient") | (jay == "out"))
    is_routine = (session_df["Duration_sec"] > 0) & (session_df["Duration_sec"] <= 4 * 3600)
    
    valid_sessions = session_df[is_outpt & is_routine].copy()
    current_patients = set(valid_sessions['Patient'].unique())
    track_patients("Base Filter: Outpatient Routine <= 4 hours", before_pats, current_patients)

    # 2. EPILEPSY DIAGNOSIS
    before_pats = current_patients.copy()
    pt_demo = clinical_df.sort_values('epilepsy_type').drop_duplicates(subset=['Patient']).set_index('Patient')
    bad_types = ["uncertain if epilepsy", "unknown or mrn not found", "non-epileptic seizure disorder", "nan", ""]
    
    epilepsy_patients = {pid for pid in current_patients if str(pt_demo.loc[pid, 'epilepsy_type']).lower().strip() not in bad_types}
    current_patients = epilepsy_patients
    track_patients("Base Filter: LLM-Confirmed Epilepsy Diagnosis", before_pats, current_patients)

    # 3. SEIZURE FREQUENCY & MATLAB IMPUTATION
    before_pats = current_patients.copy()
    allowable_visits = {
        "CONSULT VISIT", "ESTABLISHED PATIENT VISIT", "FOLLOW-UP PATIENT CLINIC", "NEW PATIENT CLINIC", 
        "NEW PATIENT VISIT", "NPV MANAGEMENT DURING COVID-19", "NPV NEUROLOGY", "RETURN ANNUAL VISIT", 
        "RETURN PATIENT EXTENDED", "RETURN PATIENT VISIT", "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN"
    }

    flat_visits = []
    for _, row in clinical_df[clinical_df['Patient'].isin(current_patients)].iterrows():
        pid = row['Patient']
        v_dates = parse_json_array(row.get('visit_dates_deid'))
        v_types = parse_json_array(row.get('visit_type'))
        v_freqs = parse_json_array(row.get('sz_freqs'), is_numeric=True, is_hassz=False)
        v_has_sz = parse_json_array(row.get('visit_hasSz'), is_numeric=True, is_hassz=True)
        
        for i in range(len(v_dates)):
            v_type = v_types[i] if i < len(v_types) else ""
            if v_type not in allowable_visits: continue 
            
            freq = v_freqs[i] if i < len(v_freqs) else np.nan
            has_sz = v_has_sz[i] if i < len(v_has_sz) else np.nan
            flat_visits.append({'Patient': pid, 'VisitDate': v_dates[i], 'Freq': freq, 'HasSz': has_sz})

    vuniq = pd.DataFrame(flat_visits)
    if not vuniq.empty:
        vuniq = vuniq.groupby(['Patient', 'VisitDate']).agg(
            Freq_R1=('Freq', lambda x: x.mean(skipna=True)),
            Has_agg=('HasSz', lambda x: x.max(skipna=True))
        ).reset_index()

        vuniq.loc[vuniq['Freq_R1'].isna() & (vuniq['Has_agg'] == 0.0), 'Freq_R1'] = 0.0
        patient_sz_freq = vuniq.groupby('Patient')['Freq_R1'].mean(skipna=True).reset_index()
        patient_sz_freq = patient_sz_freq.dropna(subset=['Freq_R1'])
        current_patients = set(patient_sz_freq['Patient'].unique())
        patient_sz_freq.rename(columns={'Freq_R1': 'mean_sz_freq'}, inplace=True)
    else:
        current_patients = set()

    track_patients("Base Filter: Documented Seizure Frequency (Primary Cohort)", before_pats, current_patients)

    # 4. CALCULATE PATIENT SPIKE RATES & AGES
    patient_spikes = valid_sessions[valid_sessions['Patient'].isin(current_patients)].groupby('Patient').agg(
        total_spikes=("count_0_46", "sum"),
        total_duration=("Duration_sec", "sum"),
        mean_age_spike=("age", "mean"),
        median_age_sz=("age", "median")
    ).reset_index()
    patient_spikes["spike_rate_per_hour"] = (patient_spikes["total_spikes"] / patient_spikes["total_duration"]) * 3600

    # 5. PROJECT SPECIFIC FILTERS & AGE DIAGNOSTICS
    print("======================================================")
    print("               PROJECT SPECIFIC FILTERS               ")
    print("======================================================\n")
    
    final_df = pd.merge(patient_sz_freq, patient_spikes, on='Patient', how='inner')
    final_df = final_df.merge(pt_demo[['nlp_gender', 'epilepsy_type', 'epilepsy_specific']].reset_index(), on='Patient', how='inner')
    
    before_pats = current_patients.copy()
    final_df = final_df[final_df['nlp_gender'].isin(['M', 'F'])]
    current_patients = set(final_df['Patient'].unique())
    track_patients("Project Filter A: Valid Gender (M or F)", before_pats, current_patients)

    before_pats = current_patients.copy()
    final_df['canonical_subtype'] = final_df.apply(assign_canonical_subtype, axis=1)
    final_df = final_df[final_df['canonical_subtype'].isin(['Focal', 'Generalized'])]
    current_patients = set(final_df['Patient'].unique())
    track_patients("Project Filter B: Focal or Generalized Subtype", before_pats, current_patients)

    # ------------------------------------------------------------------
    # DIAGNOSTIC CHECK 1: Identify & Remove Missing Age (NaN)
    # ------------------------------------------------------------------
    before_pats = current_patients.copy()
    missing_age_mask = final_df['mean_age_spike'].isna() | final_df['median_age_sz'].isna()
    missing_age_patients = final_df[missing_age_mask]['Patient'].tolist()
    
    print("-" * 60)
    if len(missing_age_patients) > 0:
        print(f"-> [DIAGNOSTIC] Found {len(missing_age_patients)} patient(s) with MISSING AGE (NaN):")
        print(f"   Patient IDs: {missing_age_patients}")
    else:
        print("-> [DIAGNOSTIC] No patients found with NaN age.")
    print("-" * 60 + "\n")
        
    final_df = final_df[~missing_age_mask]
    current_patients = set(final_df['Patient'].unique())
    track_patients("Project Filter C1: Remove Missing Age (NaN)", before_pats, current_patients)

    # ------------------------------------------------------------------
    # DIAGNOSTIC CHECK 2: Identify & Remove Pediatric / Invalid (< 18.0)
    # ------------------------------------------------------------------
    before_pats = current_patients.copy()
    invalid_age_mask = (final_df['mean_age_spike'] < 18.0) | (final_df['median_age_sz'] < 18.0)
    invalid_age_patients = final_df[invalid_age_mask]['Patient'].tolist()
    
    print("-" * 60)
    if len(invalid_age_patients) > 0:
        print(f"-> [DIAGNOSTIC] Found {len(invalid_age_patients)} patient(s) with AGE < 18 YEARS (Pediatric/Invalid):")
        for pid in invalid_age_patients:
            row = final_df[final_df['Patient'] == pid].iloc[0]
            print(f"   Patient ID: {pid} | mean_age_spike: {row['mean_age_spike']:.2f} | median_age_sz: {row['median_age_sz']:.2f}")
    else:
        print("-> [DIAGNOSTIC] No patients found with age < 18 years.")
    print("-" * 60 + "\n")
        
    final_df = final_df[~invalid_age_mask]
    current_patients = set(final_df['Patient'].unique())
    track_patients("Project Filter C2: Keep Valid Adult Age (>= 18 years)", before_pats, current_patients)

    print("======================================================")
    print(f"FINAL COHORT SIZE FOR ANALYSIS: {len(current_patients)} patients")
    print("======================================================\n")

    # ------------------------------------------------------------------
    # DIAGNOSTIC CHECK 3: Multiple EEGs Count
    # ------------------------------------------------------------------
    final_sessions = valid_sessions[valid_sessions['Patient'].isin(current_patients)]
    eeg_counts = final_sessions.groupby('Patient').size()
    multi_eeg_count = (eeg_counts > 1).sum()
    pct_multi = (multi_eeg_count / len(current_patients)) * 100
    
    print("-" * 60)
    print(f"-> [DIAGNOSTIC] Patients with >1 EEG: {multi_eeg_count} out of {len(current_patients)} ({pct_multi:.1f}%)")
    print("-" * 60 + "\n")
    
    two_multi_eeg_count = (eeg_counts > 2).sum()
    print(f"-> [DIAGNOSTIC] Patients with >2 EEGs: {two_multi_eeg_count} out of {len(current_patients)} ({(two_multi_eeg_count / len(current_patients)) * 100:.1f}%)")
    
    # 6. RUN ANALYSIS & SCATTER PLOTS 
    run_chi_square_analysis(final_df)
    
    # Print Descriptive Statistics (Medians, Quartiles, IQRs, Min/Max)
    print_descriptive_stats(final_df, "spike_rate_per_hour", "Spike Rate (per hour)")
    print_descriptive_stats(final_df, "mean_sz_freq", "Average Seizure Frequency")
    
    # Run Mann-Whitney U Tests
    run_statistical_analysis(final_df, target_var="spike_rate_per_hour", target_name="Spike Rate (per hour)")
    run_statistical_analysis(final_df, target_var="mean_sz_freq", target_name="Average Seizure Frequency")

    print("\n--- Generating Scatter Plots ---")
    plot_scatter_with_stats(final_df, "spike_rate_per_hour", "\nOverall Male vs Female: Spike Rate", "Spikes Per Hour", os.path.join(save_dir, "Figure1_Overall_Spike_Rate.png"))
    plot_scatter_with_stats(final_df, "mean_sz_freq", "\nOverall Male vs Female: Seizure Frequency", "Average Seizure Frequency", os.path.join(save_dir, "Figure2_Overall_Seizure_Frequency.png"))
    plot_stratified_scatter_with_stats(final_df, "spike_rate_per_hour", "Figure 3: Spike Rate by Sex (Focal vs Generalized)", "Spikes Per Hour", os.path.join(save_dir, "Figure3_Spike_Rate_Stratified.png"))
    plot_stratified_scatter_with_stats(final_df, "mean_sz_freq", "Figure 4: Seizure Frequency by Sex (Focal vs Generalized)", "Average Seizure Frequency", os.path.join(save_dir, "Figure4_Seizure_Frequency_Stratified.png"))

    print("\nScatter Plot Pipeline Complete.")

if __name__ == "__main__":
    main()