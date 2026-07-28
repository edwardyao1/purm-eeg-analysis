import pandas as pd
import numpy as np
import os
import ast
import warnings
import matplotlib.pyplot as plt

# Keep the console clean
warnings.filterwarnings("ignore")
np.random.seed(42)

# ==========================================================
# 1. TRACKING, PARSING & SUBTYPE HELPERS (Perfect Cohort)
# ==========================================================
def track_patients(step_name, before_pats, after_pats):
    """Prints a clean readout of patient attrition."""
    dropped = len(before_pats) - len(after_pats)
    print(f"--- {step_name} ---")
    print(f"  Patients Before: {len(before_pats)}")
    print(f"  Patients After:  {len(after_pats)}")
    print(f"  Total Dropped:   {dropped}\n")

def parse_json_array(val, is_numeric=False, is_hassz=False):
    """Safely extracts JSON arrays from CSV cells, matching MATLAB NaN logic."""
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
    if etype == "general":
        return "General"
    return np.nan


# ==========================================================
# 6. RENDER THE PRETTY TABLE
# ==========================================================
def pop_up_pretty_table(df, save_path):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.axis('off') 
    
    table = ax.table(
        cellText=df.values, 
        colLabels=df.columns, 
        cellLoc='center', 
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.2) 
    
    # Styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#348abd') 
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f8f9fa')
            else:
                cell.set_facecolor('#ffffff')
                
    # Align the first column (Relationship) to the left for better readability
    for row in range(1, len(df) + 1):
        table[(row, 0)].get_text().set_horizontalalignment('left')
                
    # UPDATED THE TITLE
    plt.title("Spearman Correlations for Variables Tested", fontweight='bold', pad=10, fontsize=15)
    
    # REMOVED THE SPEARMAN NOTE
    notes = (
        "* Sex coded as Female = 0, Male = 1\n"
        "* Epilepsy Subtype coded as Focal = 0, General = 1"
    )
    
    # MOVED THE TEXT LEFT: Changed the first number (x-coordinate) from 0.15 to 0.10
    plt.figtext(0.10, 0.09, notes, fontsize=10, color='gray')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved pretty table to: {save_path}")
    plt.show()


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

    print("=" * 75)
    print("DATA CLEANING & COHORT ATTRITION")
    print("=" * 75)

    current_patients = set(clinical_df['Patient'].dropna().unique())
    print(f"Starting Cohort: {len(current_patients)} unique patients\n")

    # 1. OUTPATIENT & ROUTINE FILTER
    before_pats = current_patients.copy()
    session_df = pd.merge(spike_df, clinical_df[['Patient', 'Session', 'acquired_on', 'report_PATIENT_CLASS', 'jay_in_or_out']], on=["Patient", "Session"], how="inner")
    
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

    # 4. CALCULATE PATIENT SPIKE RATES
    patient_spikes = valid_sessions[valid_sessions['Patient'].isin(current_patients)].groupby('Patient').agg(
        total_spikes=("count_0_46", "sum"),
        total_duration=("Duration_sec", "sum")
    ).reset_index()
    patient_spikes["spike_rate_per_hour"] = (patient_spikes["total_spikes"] / patient_spikes["total_duration"]) * 3600

    # 5. PROJECT SPECIFIC FILTERS & FINAL MERGE
    patient_df = pd.merge(patient_sz_freq, patient_spikes, on='Patient', how='inner')
    patient_df = patient_df.merge(pt_demo[['nlp_gender', 'epilepsy_type', 'epilepsy_specific']].reset_index(), on='Patient', how='inner')
    
    before_pats = current_patients.copy()
    patient_df = patient_df[patient_df['nlp_gender'].isin(['M', 'F'])]
    current_patients = set(patient_df['Patient'].unique())
    track_patients("Project Filter A: Valid Gender (M or F)", before_pats, current_patients)

    before_pats = current_patients.copy()
    patient_df['canonical_subtype'] = patient_df.apply(assign_canonical_subtype, axis=1)
    patient_df = patient_df[patient_df['canonical_subtype'].isin(['Focal', 'General'])]
    current_patients = set(patient_df['Patient'].unique())
    track_patients("Project Filter B: Focal or General Subtype", before_pats, current_patients)

    print("=" * 75)
    print(f"FINAL PATIENT COHORT SIZE: {len(current_patients):,} patients")
    print("=" * 75)

    # ==========================================================
    # 5. CALCULATE EXACT CORRELATIONS REQUESTED
    # ==========================================================
    print("=" * 75)
    print("GENERATING CORRELATION TABLE")
    print("=" * 75)

    # Prepare data
    df_corr = patient_df.copy()

    # Convert categorical to binary for correlation math
    # Sex: Female = 0, Male = 1 
    df_corr['Sex_Binary'] = (df_corr['nlp_gender'] == 'M').astype(int)

    # Epilepsy Type: Focal = 0, General = 1 (Using canonical_subtype)
    df_corr['Type_Binary'] = (df_corr['canonical_subtype'] == 'General').astype(int)

    N = len(df_corr)

    # Calculate correlations using Spearman to handle skewed seizure/spike data
    r_sex_spike  = df_corr['Sex_Binary'].corr(df_corr['spike_rate_per_hour'], method='spearman')
    r_sex_sz     = df_corr['Sex_Binary'].corr(df_corr['mean_sz_freq'], method='spearman')
    r_type_spike = df_corr['Type_Binary'].corr(df_corr['spike_rate_per_hour'], method='spearman')
    r_type_sz    = df_corr['Type_Binary'].corr(df_corr['mean_sz_freq'], method='spearman')
    r_spike_sz   = df_corr['spike_rate_per_hour'].corr(df_corr['mean_sz_freq'], method='spearman')
    r_sex_type   = df_corr['Sex_Binary'].corr(df_corr['Type_Binary'], method='spearman') # Binary vs Binary (Phi)

    # Assemble the results in the EXACT order requested
    results = [
        {"Relationship Tested": "Sex vs. Spike Rate", "N": N, "Correlation": f"{r_sex_spike:.4f}"},
        {"Relationship Tested": "Sex vs. Seizure Frequency", "N": N, "Correlation": f"{r_sex_sz:.4f}"},
        {"Relationship Tested": "Epilepsy Subtype vs. Spike Rate", "N": N, "Correlation": f"{r_type_spike:.4f}"},
        {"Relationship Tested": "Epilepsy Subtype vs. Seizure Frequency", "N": N, "Correlation": f"{r_type_sz:.4f}"},
        {"Relationship Tested": "Spike Rate vs. Seizure Frequency", "N": N, "Correlation": f"{r_spike_sz:.4f}"},
        {"Relationship Tested": "Sex vs. Epilepsy Subtype", "N": N, "Correlation": f"{r_sex_type:.4f}"}
    ]

    corr_table = pd.DataFrame(results)

    # Run the pop-up function
    table_save_path = os.path.join(save_dir, "primary_correlations_table.png")
    pop_up_pretty_table(corr_table, table_save_path)

if __name__ == "__main__":
    main()