import pandas as pd
import numpy as np
import json
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def parse_seizure_frequencies(freq_string):
    """
    Parses the stringified JSON list of seizure frequencies, removes nulls, 
    and returns a list of valid numeric frequencies.
    """
    if pd.isna(freq_string):
        return []
    try:
        # Safely load the JSON array string
        parsed_list = json.loads(freq_string)
        # Filter out None (null) values and ensure they are floats
        return [float(x) for x in parsed_list if x is not None]
    except (json.JSONDecodeError, TypeError):
        return []

def run_mwu_and_effect_size(data, group_col, val_col, group1, group2):
    """
    Runs the Mann-Whitney U test and calculates Rank-Biserial Correlation effect size.
    """
    d1 = data[data[group_col] == group1][val_col]
    d2 = data[data[group_col] == group2][val_col]
    n1, n2 = len(d1), len(d2)
    
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, n1, n2
        
    stat, pval = mannwhitneyu(d1, d2, alternative='two-sided')
    effect_size = 1 - (2 * stat) / (n1 * n2) 
    
    return stat, pval, effect_size, n1, n2

def main():
    # ---------------------------------------------------------
    # STEP 1: Load Merged Data
    # ---------------------------------------------------------
    merged_df = pd.read_csv('merged_spike_clinical_data.csv')

    # ---------------------------------------------------------
    # STEP 2: Aggregate to Patient Level (CRITICAL FOR MERGED FILE)
    # ---------------------------------------------------------
    # Because patients have multiple rows (sessions) in the merged file,
    # their clinical string array is duplicated. We use 'first' to just
    # grab the array once per unique patient.
    patient_df = merged_df.groupby('Patient').agg(
        sz_freqs=('sz_freqs', 'first'),
        nlp_gender=('nlp_gender', 'first'),
        epilepsy_type=('epilepsy_type', 'first')
    ).reset_index()

    # ---------------------------------------------------------
    # STEP 3: Parse Arrays and Calculate Mean
    # ---------------------------------------------------------
    patient_df['valid_sz_freqs'] = patient_df['sz_freqs'].apply(parse_seizure_frequencies)
    
    # Calculate the mean seizure frequency for each patient
    patient_df['mean_sz_freq'] = patient_df['valid_sz_freqs'].apply(
        lambda x: np.mean(x) if len(x) > 0 else np.nan
    )

    # Drop missing genders and patients with no valid seizure frequencies
    patient_df = patient_df.dropna(subset=['mean_sz_freq', 'nlp_gender'])
    patient_df = patient_df[patient_df['nlp_gender'].isin(['M', 'F'])]

    # =========================================================
    # At this point, `patient_df` contains ALL patients who have 
    # seizure frequency data, including Unclassified/Unknown subtypes.
    # =========================================================

    # ---------------------------------------------------------
    # STEP 4: Entire Cohort Comparison 
    # ---------------------------------------------------------
    print("=== OVERALL COHORT: SEIZURE FREQUENCY ===")
    stat, pval, es, n_m, n_f = run_mwu_and_effect_size(patient_df, 'nlp_gender', 'mean_sz_freq', 'M', 'F')
    print(f"Groups: Male (N={n_m}) vs Female (N={n_f})")
    print(f"Mann-Whitney U Stat: {stat:.2f}")
    print(f"p-value: {pval:.5f}")
    print(f"Effect Size (Rank-Biserial): {es:.4f}\n")

    # ---------------------------------------------------------
    # STEP 5: Epilepsy Subtype Comparisons (Stratified)
    # ---------------------------------------------------------
    print("=== EPILEPSY SUBTYPES: SEIZURE FREQUENCY ===")
    subtypes = ['Focal', 'General', 'Combined Generalized and Focal']
    
    results = []
    pvals_raw = []

    for subtype in subtypes:
        # Filter OUT the unknown subtypes for the sub-analyses
        sub_df = patient_df[patient_df['epilepsy_type'] == subtype]
        
        s_stat, s_pval, s_es, s_nm, s_nf = run_mwu_and_effect_size(sub_df, 'nlp_gender', 'mean_sz_freq', 'M', 'F')
        
        # Save results, appending 1.0 if the test couldn't run (e.g. empty groups)
        if not np.isnan(s_pval):
            pvals_raw.append(s_pval)
        else:
            pvals_raw.append(1.0)
            
        results.append({
            'Subtype': subtype,
            'N_Male': s_nm,
            'N_Female': s_nf,
            'U_Stat': s_stat,
            'p_value_raw': s_pval,
            'Effect_Size': s_es
        })

    # ---------------------------------------------------------
    # STEP 6: Bonferroni Correction for Subtypes
    # ---------------------------------------------------------
    # We pass the 3 raw p-values into the multitest function
    reject, pvals_corrected, _, _ = multipletests(pvals_raw, alpha=0.05, method='bonferroni')

    for res, p_corr, is_rejected in zip(results, pvals_corrected, reject):
        print(f"Subtype: {res['Subtype']}")
        print(f"  Male (N={res['N_Male']}) vs Female (N={res['N_Female']})")
        print(f"  Raw p-value:        {res['p_value_raw']:.5f}")
        print(f"  Bonferroni p-value: {p_corr:.5f} {'(Significant)' if is_rejected else '(Not Significant)'}")
        print(f"  Effect Size:        {res['Effect_Size']:.4f}\n")

if __name__ == "__main__":
    main()