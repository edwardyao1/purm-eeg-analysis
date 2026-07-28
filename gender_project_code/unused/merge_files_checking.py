import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def run_mwu_and_effect_size(data, group_col, val_col, group1, group2):
    """
    Helper function to run the Mann-Whitney U test and calculate effect size.
    Effect size metric: Rank-Biserial Correlation (r = 1 - (2U / (n1 * n2)))
    """
    # Isolate data arrays for each group
    d1 = data[data[group_col] == group1][val_col]
    d2 = data[data[group_col] == group2][val_col]
    n1, n2 = len(d1), len(d2)
    
    # Make sure there are males and females
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, n1, n2
    
    # Run Two-Sided Mann-Whitney U test
    stat, pval = mannwhitneyu(d1, d2, alternative='two-sided')
    
    # Calculate Effect Size (Rank-Biserial Correlation)
    effect_size = 1 - (2 * stat) / (n1 * n2) 
    
    return stat, pval, effect_size, n1, n2

def main():
    # ---------------------------------------------------------
    # STEP 1: Load Merged Data
    # ---------------------------------------------------------
    merged_df = pd.read_csv('merged_spike_clinical_data.csv')

    # ---------------------------------------------------------
    # STEP 2: Aggregate by Patient
    # ---------------------------------------------------------
    # Because patients have multiple EEG sessions in the merged file, 
    # we must sum their spikes/durations to get a single rate per patient.
    # We grab 'first' for gender and epilepsy type since those don't change per session.
    patient_df = merged_df.groupby('Patient').agg(
        total_spikes=('count_0_46', 'sum'),
        total_duration=('Duration_sec', 'sum'),
        nlp_gender=('nlp_gender', 'first'),
        epilepsy_type=('epilepsy_type', 'first')
    ).reset_index()

    # Calculate overall spike rate per hour
    patient_df['spike_rate_per_hour'] = (patient_df['total_spikes'] / patient_df['total_duration']) * 3600

    # Drop missing genders and keep only M/F for this specific comparison
    patient_df = patient_df.dropna(subset=['nlp_gender'])
    patient_df = patient_df[patient_df['nlp_gender'].isin(['M', 'F'])]
    
    # =========================================================
    # CRITICAL CONFIRMATION: 
    # At this stage, `patient_df` contains ALL patients (including 
    # Unclassified, Unknown, etc.) because we haven't filtered by `epilepsy_type`.
    # =========================================================

    # ---------------------------------------------------------
    # STEP 3: Entire Cohort Comparison 
    # (Includes patients without a known epilepsy subtype)
    # ---------------------------------------------------------
    print("=== ENTIRE COHORT RESULTS ===")
    stat, pval, es, n_m, n_f = run_mwu_and_effect_size(patient_df, 'nlp_gender', 'spike_rate_per_hour', 'M', 'F')
    print(f"Groups: Male (N={n_m}) vs Female (N={n_f})")
    print(f"Mann-Whitney U Stat: {stat:.2f}")
    print(f"p-value: {pval:.5f}")
    print(f"Effect Size (Rank-Biserial): {es:.4f}\n")

    # ---------------------------------------------------------
    # STEP 4: Epilepsy Subtype Comparisons (3 MWUs)
    # ---------------------------------------------------------
    print("=== SUBTYPE RESULTS ===")
    subtypes = ['Focal', 'General', 'Combined Generalized and Focal']
    
    results = []
    pvals_raw = []

    for subtype in subtypes:
        # Here is where we filter OUT the unknown subtypes for the sub-analyses
        sub_df = patient_df[patient_df['epilepsy_type'] == subtype]
        
        s_stat, s_pval, s_es, s_nm, s_nf = run_mwu_and_effect_size(sub_df, 'nlp_gender', 'spike_rate_per_hour', 'M', 'F')
        
        pvals_raw.append(s_pval)
        results.append({
            'Subtype': subtype,
            'N_Male': s_nm,
            'N_Female': s_nf,
            'U_Stat': s_stat,
            'p_value_raw': s_pval,
            'Effect_Size': s_es
        })

    # ---------------------------------------------------------
    # STEP 5: Apply Bonferroni Correction
    # ---------------------------------------------------------
    # We pass the 3 raw p-values into the multitest function
    reject, pvals_corrected, _, _ = multipletests(pvals_raw, alpha=0.05, method='bonferroni')

    # Output the corrected results
    for res, p_corr, is_rejected in zip(results, pvals_corrected, reject):
        print(f"Subtype: {res['Subtype']}")
        print(f"  Male (N={res['N_Male']}) vs Female (N={res['N_Female']})")
        print(f"  Raw p-value:        {res['p_value_raw']:.5f}")
        print(f"  Bonferroni p-value: {p_corr:.5f} {'(Significant)' if is_rejected else '(Not Significant)'}")
        print(f"  Effect Size:        {res['Effect_Size']:.4f}\n")

if __name__ == "__main__":
    main()