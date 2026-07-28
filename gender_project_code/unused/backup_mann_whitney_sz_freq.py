import pandas as pd
import numpy as np
import json
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def parse_seizure_frequencies(freq_string):
    """
    Parses the stringified list of seizure frequencies, removes nulls, 
    and returns a list of valid numeric frequencies.
    Example: "[2.5, null, 1]" -> [2.5, 1.0]
    """
    if pd.isna(freq_string):
        return []
    try:
        # The strings are formatted as JSON arrays, so json.loads is the safest parser
        # It handles 'null' natively by converting it to Python's 'None'
        parsed_list = json.loads(freq_string)
        
        # Filter out None values and ensure they are floats
        return [float(x) for x in parsed_list if x is not None]
    except (json.JSONDecodeError, TypeError):
        return []

def run_mwu_and_effect_size(data, group_col, val_col, group1, group2):
    """
    Helper function to run the Mann-Whitney U test and calculate effect size.
    Effect size metric: Rank-Biserial Correlation (r = 1 - (2U / (n1 * n2)))
    """
    d1 = data[data[group_col] == group1][val_col]
    d2 = data[data[group_col] == group2][val_col]
    n1, n2 = len(d1), len(d2)
    
    # Need at least 1 observation in each group to run a test
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, n1, n2
        
    stat, pval = mannwhitneyu(d1, d2, alternative='two-sided')
    effect_size = 1 - (2 * stat) / (n1 * n2) 
    
    return stat, pval, effect_size, n1, n2

def main():
    # ---------------------------------------------------------
    # STEP 1: Load Clinical Data Only
    # ---------------------------------------------------------
    clin_df = pd.read_csv('/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv')
    

    # ---------------------------------------------------------
    # STEP 2: Extract & Clean Seizure Frequencies
    # ---------------------------------------------------------
    # Parse the strings into actual lists of numbers
    clin_df['valid_sz_freqs'] = clin_df['sz_freqs'].apply(parse_seizure_frequencies)

    # ---------------------------------------------------------
    # STEP 3: Aggregate to Patient Level
    # ---------------------------------------------------------
    # We group by patient ID to ensure we only have 1 row per patient.
    # We combine all their recorded seizure frequencies into one pool, then take the mean.
    patient_records = []
    
    grouped = clin_df.groupby('patient_id')
    for patient_id, group in grouped:
        # Combine all frequency lists across all rows for this patient
        all_freqs_for_patient = []
        for freqs in group['valid_sz_freqs']:
            all_freqs_for_patient.extend(freqs)
            
        # Only include patients who actually have at least one valid seizure frequency recorded
        if len(all_freqs_for_patient) > 0:
            mean_sz_freq = np.mean(all_freqs_for_patient)
            
            # Grab demographic info from the first row of this patient's group
            gender = group['nlp_gender'].iloc[0]
            epi_type = group['epilepsy_type'].iloc[0]
            
            patient_records.append({
                'patient_id': patient_id,
                'mean_sz_freq': mean_sz_freq,
                'nlp_gender': gender,
                'epilepsy_type': epi_type
            })
            
    patient_df = pd.DataFrame(patient_records)

    # Clean gender column for the M/F comparison
    patient_df = patient_df.dropna(subset=['nlp_gender'])
    patient_df = patient_df[patient_df['nlp_gender'].isin(['M', 'F'])]

    # ---------------------------------------------------------
    # STEP 4: Entire Cohort Comparison 
    # (Naturally includes unknown/unclassified epilepsy subtypes)
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
        # Filter for the specific subtype (this removes unknown subtypes for these 3 tests)
        sub_df = patient_df[patient_df['epilepsy_type'] == subtype]
        
        s_stat, s_pval, s_es, s_nm, s_nf = run_mwu_and_effect_size(sub_df, 'nlp_gender', 'mean_sz_freq', 'M', 'F')
        
        # Save results, but handle case where a group might be completely empty
        if not np.isnan(s_pval):
            pvals_raw.append(s_pval)
        else:
            pvals_raw.append(1.0) # Dummy value if test couldn't run
            
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
    reject, pvals_corrected, _, _ = multipletests(pvals_raw, alpha=0.05, method='bonferroni')

    for res, p_corr, is_rejected in zip(results, pvals_corrected, reject):
        print(f"Subtype: {res['Subtype']}")
        print(f"  Male (N={res['N_Male']}) vs Female (N={res['N_Female']})")
        print(f"  Raw p-value:        {res['p_value_raw']:.5f}")
        print(f"  Bonferroni p-value: {p_corr:.5f} {'(Significant)' if is_rejected else '(Not Significant)'}")
        print(f"  Effect Size:        {res['Effect_Size']:.4f}\n")

if __name__ == "__main__":
    main()