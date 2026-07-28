import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def run_mwu_and_effect_size(data, group_col, val_col, group1, group2):
    """
    Helper function to run Mann-Whitney U test and calculate effect size.
    Uses Rank-Biserial Correlation for effect size: r = 1 - (2U / (n1 * n2))
    """
    # Isolate data for each group
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
    # STEP 1: Load Data
    # ---------------------------------------------------------
    spike_df = pd.read_csv('/Users/edwardyao/Documents/PURM/data/spike_counts.csv')
    clin_df = pd.read_csv('/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv')

    # Drop duplicates in clinical data so we have 1 row per unique patient
    clin_patients = clin_df[['patient_id', 'nlp_gender', 'epilepsy_type']].drop_duplicates(subset=['patient_id']).dropna(subset=['patient_id'])
    
    # save file and see if duplicates are gone
    # clin_patients.to_csv('/Users/edwardyao/Documents/PURM/data/clin_patients_cleaned.csv', index=False)

    # ---------------------------------------------------------
    # STEP 2: Calculate Spike Rate
    # ---------------------------------------------------------
    # Aggregate total spikes by using the 0.46 threshold and total duration per patient 
    spike_agg = spike_df.groupby('Patient').agg(
        total_spikes=('count_0_46', 'sum'),
        total_duration=('Duration_sec', 'sum')
    ).reset_index()

    # Calculate spike rate (spikes per hour)
    spike_agg['spike_rate_per_hour'] = (spike_agg['total_spikes'] / spike_agg['total_duration']) * 3600

    # ---------------------------------------------------------
    # STEP 3: Merge Dataframes
    # ---------------------------------------------------------
    # Merge the dataframs with 'Patient' in spike data -> 'patient_id' in clinical data
    df = pd.merge(spike_agg, clin_patients, left_on='Patient', right_on='patient_id', how='inner')

    # Filter out records where gender is missing or non-binary for this specific statistical comparison
    df = df.dropna(subset=['nlp_gender'])
    df = df[df['nlp_gender'].isin(['M', 'F'])]
    
    # Filter out records where gender is missing or non-binary for this specific statistical comparison
    df = df.dropna(subset=['nlp_gender'])
    df = df[df['nlp_gender'].isin(['M', 'F'])]

    # Filter out records where the spike rate could not be calculated (missing/NaN)
    df = df.dropna(subset=['spike_rate_per_hour'])
    
    # Filter out records where the patient had exactly 0 spikes
    df = df[df['spike_rate_per_hour'] > 0]

    # ---------------------------------------------------------
    # STEP 4: Entire Cohort Comparison
    # ---------------------------------------------------------
    print("=== OVERALL COHORT ===")
    stat, pval, es, n_m, n_f = run_mwu_and_effect_size(df, 'nlp_gender', 'spike_rate_per_hour', 'M', 'F')
    print(f"Groups: Male (N={n_m}) vs Female (N={n_f})")
    print(f"Mann-Whitney U Stat: {stat:.2f}")
    print(f"p-value: {pval:.5f}")
    print(f"Effect Size (Rank-Biserial): {es:.4f}\n")

    # ---------------------------------------------------------
    # STEP 5: Epilepsy Subtype Comparisons (with Bonferroni)
    # ---------------------------------------------------------
    print("=== EPILEPSY SUBTYPES ===")
    # Define the 3 major subtypes from your dataset
    subtypes = ['Focal', 'General', 'Combined Generalized and Focal']
    
    results = []
    pvals_raw = []

    for subtype in subtypes:
        # Filter for specific subtype
        sub_df = df[df['epilepsy_type'] == subtype]
        
        # Run Mann Whitney U test (MWU) and calculate effect size
        s_stat, s_pval, s_es, s_nm, s_nf = run_mwu_and_effect_size(sub_df, 'nlp_gender', 'spike_rate_per_hour', 'M', 'F')
        print(f"Groups: Male (N={s_nm}) vs Female (N={s_nf})")
    
        pvals_raw.append(s_pval)
        results.append({
            'Subtype': subtype,
            'N_Male': s_nm,
            'N_Female': s_nf,
            'U_Stat': s_stat,
            'p_value_raw': s_pval,
            'Effect_Size': s_es
        })

    # Apply Bonferroni Correction using statsmodels
    # alpha = 0.05 is the desired family-wise error rate
    reject, pvals_corrected, _, _ = multipletests(pvals_raw, alpha=0.05, method='bonferroni')

    # ---------------------------------------------------------
    # STEP 6: Output the Subtype Results
    # ---------------------------------------------------------
    for res, p_corr, is_rejected in zip(results, pvals_corrected, reject):
        res['p_value_bonferroni'] = p_corr
        res['Significant'] = is_rejected
        
        print(f"Subtype: {res['Subtype']}")
        print(f"  Male (N={res['N_Male']}) vs Female (N={res['N_Female']})")
        print(f"  Raw p-value:        {res['p_value_raw']:.5f}")
        print(f"  Bonferroni p-value: {p_corr:.5f} {'(Significant)' if is_rejected else '(Not Significant)'}")
        print(f"  Effect Size:        {res['Effect_Size']:.4f}\n")

    # Save to a clean CSV for reference
    res_df = pd.DataFrame(results)
    res_df.to_csv('spike_rate_sex_comparison_results.csv', index=False)
    print("Saved detailed results to 'spike_rate_sex_comparison_results.csv'")

if __name__ == "__main__":
    main()