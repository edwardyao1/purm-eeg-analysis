import pandas as pd
import numpy as np
import os
import ast
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

# ==========================================================
# TRACKING, PARSING & SUBTYPE HELPERS (Original Pipeline)
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
# LOGISTIC BOOTSTRAP HELPERS
# ==========================================================
def bootstrap_regression_coeffs(df, outcome_col, formula_vars, n_boot=5000):
    df = df.copy()

    print(f"\nChecking {outcome_col}")
    print(df[outcome_col].value_counts())

    df = df[np.isfinite(df[outcome_col])]

    # Get dummies. Note: We use canonical_subtype to maintain the exact cohort.
    reg_df = pd.get_dummies(
        df[[outcome_col, "nlp_gender", "canonical_subtype"]],
        drop_first=True
    ).astype(float)

    results = {}

    for var in formula_vars:
        if var not in reg_df.columns:
            print(f"{var} missing")
            results[var] = (0, 0, 0, 1)
            continue

        coeffs = []
        for _ in range(n_boot):
            sample = reg_df.sample(len(reg_df), replace=True)
            X = sample[formula_vars]
            X = sm.add_constant(X)
            y = sample[outcome_col]

            try:
                # Using Logistic Regression for binary outcomes
                model = sm.Logit(y, X).fit(disp=0)
                beta = model.params[var]

                if np.isfinite(beta):
                    coeffs.append(beta)
            except Exception:
                # Logit might fail to converge on weird samples; pass safely
                pass

        coeffs = np.asarray(coeffs)
        print(f"{outcome_col} | {var} | valid coeffs = {len(coeffs)}")

        if len(coeffs) == 0:
            results[var] = (0, 0, 0, 1)
            continue

        p = 2 * min(np.mean(coeffs >= 0), np.mean(coeffs <= 0))
        results[var] = (
            float(np.mean(coeffs)),
            float(np.percentile(coeffs, 2.5)),
            float(np.percentile(coeffs, 97.5)),
            float(p)
        )

    return results

def run_analysis(patient_df, outcome_col, outcome_label):
    print(f"\n===========================================================================")
    print(f"BOOTSTRAP LOGIT REGRESSION — {outcome_label}")
    print(f"===========================================================================")
    
    formula_vars = ['nlp_gender_M', 'canonical_subtype_General']
    coeffs = bootstrap_regression_coeffs(patient_df, outcome_col, formula_vars)
    
    rows = []
    
    # Section 1: Sex
    m_res = coeffs['nlp_gender_M']
    female_n = len(patient_df[patient_df['nlp_gender']=='F'])
    rows.append(dict(section="Sex", label="Female", n=female_n, is_reference=True, diff=0, lo=0, hi=0, p=None))
    rows.append(dict(section="Sex", label="Male", n=len(patient_df[patient_df['nlp_gender']=='M']), is_reference=False, diff=m_res[0], lo=m_res[1], hi=m_res[2], p=m_res[3]))

    # Section 2: Epilepsy
    g_res = coeffs['canonical_subtype_General']
    focal_n = len(patient_df[patient_df['canonical_subtype']=='Focal'])
    rows.append(dict(section="Epilepsy Type", label="Focal", n=focal_n, is_reference=True, diff=0, lo=0, hi=0, p=None))
    rows.append(dict(section="Epilepsy Type", label="General", n=len(patient_df[patient_df['canonical_subtype']=='General']), is_reference=False, diff=g_res[0], lo=g_res[1], hi=g_res[2], p=g_res[3]))

    return rows


# ==========================================================
# PLOTTING HELPERS
# ==========================================================
def forest_plot(rows, title, x_lim, x_ticks, x_label, out_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    sections = []
    for r in rows:
        if r["section"] not in sections:
            sections.append(r["section"])

    ypos = {}
    y = len(rows) + len(sections) + 2

    for sec in sections:
        ypos[(sec, "__header__")] = y
        y -= 1
        for r in rows:
            if r["section"] == sec:
                ypos[(sec, r["label"])] = y
                y -= 1
        y -= 0.5

    COL_SUB = 0.02
    COL_N = 0.22 
    COL_EST = 0.38
    PLOT_L = 0.52   
    PLOT_R = 0.90
    COL_P = 0.93

    def map_x(v):
        frac = (v - x_lim[0]) / (x_lim[1] - x_lim[0])
        return PLOT_L + frac * (PLOT_R - PLOT_L)

    top = max(ypos.values()) + 1.5

    ax.text(COL_SUB, top, "Subgroup", fontweight="bold")
    ax.text(COL_N, top, "No. of Patients", fontweight="bold")
    ax.text(COL_EST, top, "Coefficient (95% CI)", fontweight="bold")
    ax.text(COL_P, top, "P-Value", fontweight="bold")
    ax.plot([0, 1], [top - 0.3, top - 0.3], color="black")

    for sec in sections:
        ax.text(COL_SUB, ypos[(sec, "__header__")], sec, fontsize=12, fontweight="bold")

    for row in rows:
        y = ypos[(row["section"], row["label"])]
        ax.text(COL_SUB + 0.04, y, row["label"])
        ax.text(COL_N, y, str(row["n"]))

        if row["is_reference"]:
            ax.text(COL_EST, y, f"Reference")
            ax.scatter(map_x(0), y, s=50, marker="s", color="black")
            ax.text(COL_P, y, "Ref.")
        else:
            d = row["diff"]
            lo = row["lo"]
            hi = row["hi"]

            ax.text(COL_EST, y, f"{d:+.2f} ({lo:+.2f}, {hi:+.2f})")
            ax.plot([map_x(lo), map_x(hi)], [y, y], lw=2, color="black")
            ax.scatter(map_x(d), y, s=50, marker="s", color="black")

            ptxt = "<0.001" if row["p"] < 0.001 else f"{row['p']:.3f}"
            ax.text(COL_P, y, ptxt)

    ax.plot([map_x(0), map_x(0)], [0, top], ls="--", color="gray")
    axis_y = -0.5
    ax.plot([map_x(x_lim[0]), map_x(x_lim[1])], [axis_y, axis_y], color="black")

    for tick in x_ticks:
        x = map_x(tick)
        ax.plot([x, x], [axis_y, axis_y - 0.2], color="black")
        ax.text(x, axis_y - 0.55, str(tick), ha="center", va="top")
        
    center_x = map_x((x_lim[0] + x_lim[1]) / 2)
    ax.text(center_x, axis_y - 1.8, x_label, ha="center", fontweight="bold", fontsize=11)

    plt.title(title, fontweight="bold", pad=20)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================================
# STANDARD LOGISTIC OLS RESULTS
# ==========================================================
def print_logit_results(df, outcome_col, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    reg_df = df.copy()

    reg_df = reg_df[["nlp_gender", "canonical_subtype", outcome_col]].copy()

    reg_df[outcome_col] = pd.to_numeric(reg_df[outcome_col], errors="coerce")
    reg_df = reg_df.dropna(subset=[outcome_col])
    reg_df = reg_df[np.isfinite(reg_df[outcome_col])].copy()

    print(f"\nBinary Outcome Summary ({outcome_col}):")
    print(reg_df[outcome_col].value_counts())

    # Convert to floats
    reg_df["nlp_gender_M"] = (reg_df["nlp_gender"] == "M").astype(float)
    reg_df["canonical_subtype_General"] = (reg_df["canonical_subtype"] == "General").astype(float)

    X = reg_df[["nlp_gender_M", "canonical_subtype_General"]].astype(float)
    X = sm.add_constant(X)
    
    y = reg_df[outcome_col].astype(float)

    # Run the Logistic Regression
    model = sm.Logit(y, X).fit()

    print(model.summary())
    print("\n95% CI (Log-Odds)")
    print(model.conf_int())
    
    print("\nOdds Ratios (Exponentiated Coefficients)")
    print(np.exp(model.params))

    return model


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

    # --- BINARY CONVERSION BASED ON MEDIANS ---
    # Median spike_rate = 1.79, Median sz_freq = 0.73 (hardcoded matching user's original logic)
    patient_df["spike_rate_binary"] = (patient_df["spike_rate_per_hour"] >= 1.79).astype(int)
    patient_df["sz_freq_binary"] = (patient_df["mean_sz_freq"] >= 0.73).astype(int)

    print("=" * 75)
    print(f"FINAL PATIENT COHORT SIZE: {len(current_patients):,} patients")
    print("=" * 75)

    # 6. RUN BOOTSTRAP ANALYSIS
    rows_spike = run_analysis(patient_df, "spike_rate_binary", "SPIKE RATE (≥1.79/hr)")
    rows_sz    = run_analysis(patient_df, "sz_freq_binary", "SEIZURE FREQUENCY (≥0.73/mo)")

    # 7. GENERATE FOREST PLOTS
    print("\n--- Generating Forest Plots ---")
    forest_plot(
        rows_spike,
        "Bootstrapped Logit Coefficients: Spike Rate (≥1.79/hr)",
        (-0.4, 0.6),
        [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6],
        "Coefficient", 
        os.path.join(save_dir, "forest_spike_rate_boot_binary.png")
    )

    forest_plot(
        rows_sz,
        "Bootstrapped Logit Coefficients: Seizure Frequency (≥0.73/mo)",
        (-0.6, 0.2),
        [-0.6, -0.4, -0.2, 0.0, 0.2],
        "Coefficient", 
        os.path.join(save_dir, "forest_seizure_freq_boot_binary.png")
    )
    print("Done plotting.")

    # 8. STANDARD LOGISTIC REGRESSION RESULTS
    spike_model = print_logit_results(patient_df, "spike_rate_binary", "LOGISTIC REGRESSION: SPIKE RATE (≥1.79/hr)")
    sz_model = print_logit_results(patient_df, "sz_freq_binary", "LOGISTIC REGRESSION: SEIZURE FREQUENCY (≥0.73/mo)")

if __name__ == "__main__":
    main()