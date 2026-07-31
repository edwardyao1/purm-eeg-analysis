# Bootstrapped binary forest plot of Logit models for spike rate and seizure frequency 
# Models:
# 1. Base (Sex + Subtype + Age >= 51)
# 2. Base + (Sex x Age >= 51)
# 3. Base + (Sex x Subtype)

import pandas as pd
import numpy as np
import os
import ast
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import chi2

warnings.filterwarnings("ignore")
np.random.seed(42)

# ==========================================================
# TRACKING, PARSING & SUBTYPE HELPERS
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
# LIKELIHOOD RATIO TEST HELPER
# ==========================================================
def perform_lrt(model_base, model_alt, alt_name):
    lr_stat = -2 * (model_base.llf - model_alt.llf)
    df_diff = model_alt.df_model - model_base.df_model
    p_val = chi2.sf(lr_stat, df_diff)
    
    print("\n" + "="*80)
    print(f"LIKELIHOOD RATIO TEST: {alt_name}")
    print("="*80)
    print(f"Base Model Log-Likelihood:        {model_base.llf:.3f}")
    print(f"Interactive Model Log-Likelihood: {model_alt.llf:.3f}")
    print(f"Likelihood Ratio Statistic:       {lr_stat:.3f}")
    print(f"Degrees of Freedom Diff:          {df_diff}")
    print(f"LRT P-value:                      {p_val:.5g}")
    if p_val < 0.05:
        print(f"CONCLUSION: YES. The interaction term significantly improves the model fit.")
    else:
        print(f"CONCLUSION: NO. The interaction term DOES NOT significantly improve the model fit.")
    print("="*80 + "\n")

# ==========================================================
# LOGISTIC BOOTSTRAP HELPERS (WITH LOCKED RANDOM SEED)
# ==========================================================
def bootstrap_regression_coeffs(df, outcome_col, formula_vars, n_boot=5000):
    df = df.copy()
    np.random.seed(42)
    df = df[np.isfinite(df[outcome_col])]
    results = {}

    for var in formula_vars:
        if var not in df.columns:
            results[var] = (0, 0, 0, 1)
            continue
        coeffs = []
        for _ in range(n_boot):
            sample = df.sample(len(df), replace=True)
            X = sample[formula_vars]
            X = sm.add_constant(X)
            y = sample[outcome_col]

            try:
                model = sm.Logit(y, X).fit(disp=0)
                beta = model.params[var]
                if np.isfinite(beta):
                    coeffs.append(beta)
            except Exception:
                pass

        coeffs = np.asarray(coeffs)
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

def run_analysis(patient_df, outcome_col, formula_vars):
    coeffs = bootstrap_regression_coeffs(patient_df, outcome_col, formula_vars)
    rows = []
    
    def to_or(res):
        return (np.exp(res[0]), np.exp(res[1]), np.exp(res[2]), res[3])

    # Section 1: Sex
    m_res = to_or(coeffs['nlp_gender_M'])
    female_n = len(patient_df[patient_df['nlp_gender']=='F'])
    rows.append(dict(section="Sex", label="Female", n=female_n, is_reference=True, diff=1.0, lo=1.0, hi=1.0, p=None))
    rows.append(dict(section="Sex", label="Male", n=len(patient_df[patient_df['nlp_gender']=='M']), is_reference=False, diff=m_res[0], lo=m_res[1], hi=m_res[2], p=m_res[3]))

    # Section 2: Epilepsy Type
    g_res = to_or(coeffs['canonical_subtype_Generalized'])
    focal_n = len(patient_df[patient_df['canonical_subtype']=='Focal'])
    rows.append(dict(section="Epilepsy Type", label="Focal", n=focal_n, is_reference=True, diff=1.0, lo=1.0, hi=1.0, p=None))
    rows.append(dict(section="Epilepsy Type", label="Generalized", n=len(patient_df[patient_df['canonical_subtype']=='Generalized']), is_reference=False, diff=g_res[0], lo=g_res[1], hi=g_res[2], p=g_res[3]))

    # Section 3: Age Group
    age_lt51_n = len(patient_df[patient_df['age_ge_51']==0])
    rows.append(dict(section="Age Group", label="< 51 years", n=age_lt51_n, is_reference=True, diff=1.0, lo=1.0, hi=1.0, p=None))
    a_res = to_or(coeffs['age_ge_51'])
    rows.append(dict(section="Age Group", label=">= 51 years", n=len(patient_df[patient_df['age_ge_51']==1]), is_reference=False, diff=a_res[0], lo=a_res[1], hi=a_res[2], p=a_res[3]))

    # Section 4: Interaction Term (Dynamic based on formula)
    if 'gender_M_x_age_ge_51' in formula_vars:
        i_res = to_or(coeffs['gender_M_x_age_ge_51'])
        rows.append(dict(section="Interaction", label="No Interaction", n=len(patient_df), is_reference=True, diff=1.0, lo=1.0, hi=1.0, p=None))
        rows.append(dict(section="Interaction", label="Male x Age >= 51", n=len(patient_df[(patient_df['nlp_gender_M']==1) & (patient_df['age_ge_51']==1)]), is_reference=False, diff=i_res[0], lo=i_res[1], hi=i_res[2], p=i_res[3]))
        
    if 'gender_M_x_subtype_Gen' in formula_vars:
        i_res = to_or(coeffs['gender_M_x_subtype_Gen'])
        rows.append(dict(section="Interaction", label="No Interaction", n=len(patient_df), is_reference=True, diff=1.0, lo=1.0, hi=1.0, p=None))
        rows.append(dict(section="Interaction", label="Male x Generalized", n=len(patient_df[(patient_df['nlp_gender_M']==1) & (patient_df['canonical_subtype_Generalized']==1)]), is_reference=False, diff=i_res[0], lo=i_res[1], hi=i_res[2], p=i_res[3]))

    return rows

# ==========================================================
# CUSTOM FOREST PLOT (VERBATIM FROM YOUR ORIGINAL SCRIPT)
# ==========================================================
def forest_plot(rows, title, x_lim, x_ticks, x_label, left_dir_label, right_dir_label, out_path):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    sections = []
    for r in rows:
        if r["section"] not in sections:
            sections.append(r["section"])

    COL_SUB = 0.02
    COL_N = 0.25   
    COL_EST = 0.41 
    PLOT_L = 0.59  
    PLOT_R = 0.86  
    COL_P = 0.91   

    def map_x(v):
        frac = (v - x_lim[0]) / (x_lim[1] - x_lim[0])
        return PLOT_L + frac * (PLOT_R - PLOT_L)

    top = 10.0
    section_gap = 0.95  
    line_gap = 0.75     
    
    y = top - section_gap
    ypos = {}

    for sec in sections:
        ypos[(sec, "__header__")] = y
        y -= line_gap
        for r in rows:
            if r["section"] == sec:
                ypos[(sec, r["label"])] = y
                y -= line_gap
        y -= (section_gap - line_gap) 

    ax.text(COL_SUB, top, "Subgroup / Variable", fontweight="bold", va="center", fontsize=14)
    ax.text(COL_N, top, "Patients (N)", fontweight="bold", va="center", fontsize=14)
    ax.text(COL_EST, top, "Odds Ratio (95% CI)", fontweight="bold", va="center", fontsize=14) 
    ax.text(COL_P, top, "P-Value", fontweight="bold", va="center", fontsize=14)
    ax.text(map_x(1.0), top, "Odds Ratio (95% CI)", fontweight="bold", ha="center", va="center", fontsize=14)

    ax.plot([0, 1], [top - 0.3, top - 0.3], color="black")

    for sec in sections:
        ax.text(COL_SUB, ypos[(sec, "__header__")], sec, fontsize=15, fontweight="bold", va="center")

    for row in rows:
        y_loc = ypos[(row["section"], row["label"])]
        label_text = f'{row["label"]} (Ref.)' if row["is_reference"] else row["label"]
        
        ax.text(COL_SUB + 0.04, y_loc, label_text, va="center", fontsize=14)
        ax.text(COL_N, y_loc, str(row["n"]), va="center", fontsize=14)

        if row["is_reference"]:
            ax.text(COL_EST, y_loc, "Ref.", va="center", fontsize=14)
            ax.scatter(map_x(1.0), y_loc, s=50, marker="s", color="black")
            ax.text(COL_P, y_loc, "Ref.", va="center", fontsize=14)
        else:
            d = row["diff"]
            lo = row["lo"]
            hi = row["hi"]

            ax.text(COL_EST, y_loc, f"{d:.2f} ({lo:.2f}-{hi:.2f})", va="center", fontsize=14)
            ax.plot([map_x(lo), map_x(hi)], [y_loc, y_loc], lw=2, color="black")
            ax.scatter(map_x(d), y_loc, s=50, marker="s", color="black")

            ptxt = "<0.001" if row["p"] < 0.001 else f"{row['p']:.3f}"
            ax.text(COL_P, y_loc, ptxt, va="center", fontsize=14)

    bottom_y = min(ypos.values())
    axis_y = bottom_y - 0.8  
    
    ax.plot([map_x(1.0), map_x(1.0)], [axis_y, top - 0.3], ls="--", color="gray")
    ax.plot([map_x(x_lim[0]), map_x(x_lim[1])], [axis_y, axis_y], color="black")

    for tick in x_ticks:
        x_loc = map_x(tick)
        ax.plot([x_loc, x_loc], [axis_y, axis_y - 0.15], color="black") 
        ax.text(x_loc, axis_y - 0.30, str(tick), ha="center", va="top", fontsize=12)

    arrow_y = axis_y - 1.0
    axis_range = x_lim[1] - x_lim[0]
    
    arrow_left = map_x(x_lim[0] + axis_range * 0.0)
    arrow_right = map_x(x_lim[0] + axis_range * 1.0)
    
    ax.annotate("", xy=(arrow_left, arrow_y), xytext=(arrow_right, arrow_y),
                arrowprops=dict(arrowstyle="<->", lw=1.8, color="black"))
    
    text_left_x = map_x(x_lim[0] + axis_range * 0.15)
    text_right_x = map_x(x_lim[0] + axis_range * 0.85)
    
    ax.text(text_left_x, arrow_y + 0.15, left_dir_label, 
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="black")
    ax.text(text_right_x, arrow_y + 0.15, right_dir_label, 
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="black")

    plt.title(title, fontweight="bold", fontsize=17, pad=25)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


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
    print("LOADING DATA & PROCESSING COHORT")
    print("=" * 75)

    clinical_df = pd.read_csv(clinical_csv, low_memory=False).rename(columns={'patient_id': 'Patient', 'session_number': 'Session'})
    spike_df = pd.read_csv(spike_csv, low_memory=False)

    clinical_df['start_time_deid'] = pd.to_datetime(clinical_df['start_time_deid'], errors='coerce')
    clinical_df['deid_birth_date'] = pd.to_datetime(clinical_df['deid_birth_date'], errors='coerce')
    clinical_df['age'] = (clinical_df['start_time_deid'] - clinical_df['deid_birth_date']).dt.days / 365.25

    current_patients = set(clinical_df['Patient'].dropna().unique())

    # 1. OUTPATIENT ROUTINE
    session_df = pd.merge(spike_df, clinical_df[['Patient', 'Session', 'acquired_on', 'report_PATIENT_CLASS', 'jay_in_or_out', 'age']], on=["Patient", "Session"], how="inner")
    acq = session_df["acquired_on"].fillna("").astype(str).str.lower()
    p_class = session_df["report_PATIENT_CLASS"].fillna("").astype(str).str.lower()
    jay = session_df["jay_in_or_out"].fillna("").astype(str).str.lower()
    is_outpt = (acq.str.contains("spe") | acq.str.contains("radnor") | (p_class == "outpatient") | (jay == "out"))
    is_routine = (session_df["Duration_sec"] > 0) & (session_df["Duration_sec"] <= 4 * 3600)
    valid_sessions = session_df[is_outpt & is_routine].copy()
    current_patients = set(valid_sessions['Patient'].unique())

    # 2. EPILEPSY DIAGNOSIS
    pt_demo = clinical_df.sort_values('epilepsy_type').drop_duplicates(subset=['Patient']).set_index('Patient')
    bad_types = ["uncertain if epilepsy", "unknown or mrn not found", "non-epileptic seizure disorder", "nan", ""]
    current_patients = {pid for pid in current_patients if str(pt_demo.loc[pid, 'epilepsy_type']).lower().strip() not in bad_types}

    # 3. SEIZURE FREQUENCY WITH MISSING IMPUTATION
    allowable_visits = {"CONSULT VISIT", "ESTABLISHED PATIENT VISIT", "FOLLOW-UP PATIENT CLINIC", "NEW PATIENT CLINIC", "NEW PATIENT VISIT", "NPV MANAGEMENT DURING COVID-19", "NPV NEUROLOGY", "RETURN ANNUAL VISIT", "RETURN PATIENT EXTENDED", "RETURN PATIENT VISIT", "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN"}
    
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
        vuniq['VisitDate'] = pd.to_datetime(vuniq['VisitDate'], errors='coerce')
        vuniq['had_doc_freq'] = vuniq['Freq'].notna()
        
        vuniq = vuniq.groupby(['Patient', 'VisitDate']).agg(
            Freq_R1=('Freq', lambda x: x.mean(skipna=True)),
            Has_agg=('HasSz', lambda x: x.max(skipna=True)),
            had_doc_freq=('had_doc_freq', 'max')
        ).reset_index()

        vuniq.loc[vuniq['Freq_R1'].isna() & (vuniq['Has_agg'] == 0.0), 'Freq_R1'] = 0.0
        patient_sz_freq = vuniq.groupby('Patient')['Freq_R1'].mean(skipna=True).reset_index()
        patient_sz_freq = patient_sz_freq.dropna(subset=['Freq_R1'])
        current_patients = set(patient_sz_freq['Patient'].unique())
        patient_sz_freq.rename(columns={'Freq_R1': 'mean_sz_freq'}, inplace=True)
    else:
        current_patients = set()

    # 4. SPIKE RATES & FILTERS
    patient_spikes = valid_sessions[valid_sessions['Patient'].isin(current_patients)].groupby('Patient').agg(
        total_spikes=("count_0_46", "sum"),
        total_duration=("Duration_sec", "sum"),
        mean_age_spike=("age", "mean"),
        median_age_sz=("age", "median")
    ).reset_index()
    patient_spikes["spike_rate_per_hour"] = (patient_spikes["total_spikes"] / patient_spikes["total_duration"]) * 3600

    patient_df = pd.merge(patient_sz_freq, patient_spikes, on='Patient', how='inner').merge(pt_demo[['nlp_gender', 'epilepsy_type', 'epilepsy_specific']].reset_index(), on='Patient', how='inner')
    
    patient_df = patient_df.dropna(subset=['spike_rate_per_hour'])
    patient_df = patient_df[patient_df['nlp_gender'].isin(['M', 'F'])]
    patient_df['canonical_subtype'] = patient_df.apply(assign_canonical_subtype, axis=1)
    patient_df = patient_df[patient_df['canonical_subtype'].isin(['Focal', 'Generalized'])]
    patient_df = patient_df[~(patient_df['mean_age_spike'].isna() | patient_df['median_age_sz'].isna())]
    patient_df = patient_df[(patient_df['mean_age_spike'] >= 18.0) & (patient_df['median_age_sz'] >= 18.0)]

    # Variables for models
    patient_df['age_ge_51'] = (patient_df['mean_age_spike'] >= 51.0).astype(int)
    patient_df["nlp_gender_M"] = (patient_df["nlp_gender"] == "M").astype(int)
    patient_df["canonical_subtype_Generalized"] = (patient_df["canonical_subtype"] == "Generalized").astype(int)
    
    # Interaction terms
    patient_df["gender_M_x_age_ge_51"] = patient_df["nlp_gender_M"] * patient_df["age_ge_51"]
    patient_df["gender_M_x_subtype_Gen"] = patient_df["nlp_gender_M"] * patient_df["canonical_subtype_Generalized"]

    med_spike = patient_df["spike_rate_per_hour"].median()
    med_sz = patient_df["mean_sz_freq"].median()
    
    patient_df["spike_rate_binary"] = (patient_df["spike_rate_per_hour"] >= med_spike).astype(int)
    patient_df["sz_freq_binary"] = (patient_df["mean_sz_freq"] >= med_sz).astype(int)

    # ---------------------------------------------------------
    # RUN ALL MODELS, LRTs, AND GENERATE FOREST PLOTS
    # ---------------------------------------------------------
    outcomes = [
        ("spike_rate_binary", f"Spike Rate (≥{med_spike:.2f}/hr)", "Spikes"),
        ("sz_freq_binary", f"Seizure Freq (≥{med_sz:.2f}/mo)", "Seizures")
    ]

    base_vars = ["nlp_gender_M", "canonical_subtype_Generalized", "age_ge_51"]
    int1_vars = ["nlp_gender_M", "canonical_subtype_Generalized", "age_ge_51", "gender_M_x_age_ge_51"]
    int2_vars = ["nlp_gender_M", "canonical_subtype_Generalized", "age_ge_51", "gender_M_x_subtype_Gen"]

    for col, title, suffix in outcomes:
        print("\n" + "=" * 80)
        print(f"PROCESSING OUTCOME: {col.upper()}")
        print("=" * 80)
        
        reg_df = patient_df.copy().dropna(subset=[col])
        y = reg_df[col].astype(float)
        
        # 1. Base Model
        print(f"\n--- 1. BASE MODEL ---")
        X_base = sm.add_constant(reg_df[base_vars])
        model_base = sm.Logit(y, X_base).fit(disp=0)
        print(model_base.summary())
        
        rows_base = run_analysis(reg_df, col, base_vars)
        forest_plot(rows_base, f"ORs: {title} ~ Sex + Type + Age", (0.0, 2.0), [0.0, 0.5, 1.0, 1.5, 2.0], "", f"← Fewer {suffix}", f"More {suffix} →", os.path.join(save_dir, f"forest_{col}_1_Base.png"))

        # 2. Interaction 1: Sex x Age
        print(f"\n--- 2. INTERACTION (SEX x AGE) ---")
        X_int1 = sm.add_constant(reg_df[int1_vars])
        model_int1 = sm.Logit(y, X_int1).fit(disp=0)
        print(model_int1.summary())
        
        perform_lrt(model_base, model_int1, f"{col} - Base vs (Sex x Age >= 51)")
        
        rows_int1 = run_analysis(reg_df, col, int1_vars)
        forest_plot(rows_int1, f"ORs: {title} ~ Sex + Type + Age + Interaction", (0.0, 2.0), [0.0, 0.5, 1.0, 1.5, 2.0], "", f"← Fewer {suffix}", f"More {suffix} →", os.path.join(save_dir, f"forest_{col}_2_Int_SexAge.png"))

        # 3. Interaction 2: Sex x Subtype
        print(f"\n--- 3. INTERACTION (SEX x SUBTYPE) ---")
        X_int2 = sm.add_constant(reg_df[int2_vars])
        model_int2 = sm.Logit(y, X_int2).fit(disp=0)
        print(model_int2.summary())
        
        perform_lrt(model_base, model_int2, f"{col} - Base vs (Sex x Subtype)")
        
        rows_int2 = run_analysis(reg_df, col, int2_vars)
        forest_plot(rows_int2, f"ORs: {title} ~ Sex + Type + Age + Interaction", (0.0, 2.0), [0.0, 0.5, 1.0, 1.5, 2.0], "", f"← Fewer {suffix}", f"More {suffix} →", os.path.join(save_dir, f"forest_{col}_3_Int_SexSubtype.png"))

    print("\n===========================================================================")
    print("ALL MODELS, LRTs, AND FOREST PLOTS GENERATED SUCCESSFULLY.")
    print(f"Check your directory at: {save_dir}")
    print("===========================================================================\n")

if __name__ == "__main__":
    main()