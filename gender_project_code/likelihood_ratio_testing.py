# Bootstrapped binary forest plot of Logit models for spike rate and seizure frequency 
# Modeled by sex, epilepsy type, and age groups (18-39, 40-64, 65+) 
# with median split for binary spike rate and seizure frequency outcomes
# INCLUDES: Likelihood Ratio Tests (LRT) for Models M1 through M5 vs M0

import pandas as pd
import numpy as np
import os
import ast
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

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

    cols_to_dummy = [outcome_col, "nlp_gender", "canonical_subtype", "age_group"]
    reg_df = pd.get_dummies(
        df[cols_to_dummy],
        drop_first=True
    ).astype(float)

    results = {}

    for var in formula_vars:
        if var not in reg_df.columns:
            print(f"{var} missing")
            results[var] = (1, 1, 1, 1) # OR=1 is the null
            continue

        coeffs = []
        for _ in range(n_boot):
            sample = reg_df.sample(len(reg_df), replace=True)
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
        print(f"{outcome_col} | {var} | valid coeffs = {len(coeffs)}")

        if len(coeffs) == 0:
            results[var] = (1, 1, 1, 1)
            continue

        p = 2 * min(np.mean(coeffs >= 0), np.mean(coeffs <= 0))
        
        or_coeffs = np.exp(coeffs)
        
        results[var] = (
            float(np.median(or_coeffs)),
            float(np.percentile(or_coeffs, 2.5)),
            float(np.percentile(or_coeffs, 97.5)),
            float(p)
        )

    return results

def run_analysis(patient_df, outcome_col, outcome_label):
    print(f"\n===========================================================================")
    print(f"BOOTSTRAP LOGIT REGRESSION (ODDS RATIOS) — {outcome_label}")
    print(f"===========================================================================")
    
    formula_vars = ['nlp_gender_M', 'canonical_subtype_General', 'age_group_40-64', 'age_group_65+']
    coeffs = bootstrap_regression_coeffs(patient_df, outcome_col, formula_vars)
    
    rows = []
    
    m_res = coeffs['nlp_gender_M']
    female_n = len(patient_df[patient_df['nlp_gender']=='F'])
    rows.append(dict(section="Sex", label="Female", n=female_n, is_reference=True, diff=1.0, lo=1.0, hi=1.0, p=None))
    rows.append(dict(section="Sex", label="Male", n=len(patient_df[patient_df['nlp_gender']=='M']), is_reference=False, diff=m_res[0], lo=m_res[1], hi=m_res[2], p=m_res[3]))

    g_res = coeffs['canonical_subtype_General']
    focal_n = len(patient_df[patient_df['canonical_subtype']=='Focal'])
    rows.append(dict(section="Epilepsy Type", label="Focal", n=focal_n, is_reference=True, diff=1.0, lo=1.0, hi=1.0, p=None))
    rows.append(dict(section="Epilepsy Type", label="General", n=len(patient_df[patient_df['canonical_subtype']=='General']), is_reference=False, diff=g_res[0], lo=g_res[1], hi=g_res[2], p=g_res[3]))

    age1_n = len(patient_df[patient_df['age_group']=='18-39'])
    rows.append(dict(section="Age Group", label="18-39 years", n=age1_n, is_reference=True, diff=1.0, lo=1.0, hi=1.0, p=None))
    
    a2_res = coeffs['age_group_40-64']
    rows.append(dict(section="Age Group", label="40-64 years", n=len(patient_df[patient_df['age_group']=='40-64']), is_reference=False, diff=a2_res[0], lo=a2_res[1], hi=a2_res[2], p=a2_res[3]))
    
    a3_res = coeffs['age_group_65+']
    rows.append(dict(section="Age Group", label="65+ years", n=len(patient_df[patient_df['age_group']=='65+']), is_reference=False, diff=a3_res[0], lo=a3_res[1], hi=a3_res[2], p=a3_res[3]))

    return rows

# ==========================================================
# PLOTTING HELPERS
# ==========================================================
def forest_plot(rows, title, x_lim, x_ticks, x_label, out_path):
    fig, ax = plt.subplots(figsize=(14, 8))
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

    ax.text(COL_SUB, top, "Subgroup / Variable", fontweight="bold", va="center")
    ax.text(COL_N, top, "No. of Patients", fontweight="bold", va="center")
    ax.text(COL_EST, top, "Odds Ratio (95% CI)", fontweight="bold", va="center")
    ax.text(COL_P, top, "P-Value", fontweight="bold", va="center")
    ax.plot([0, 1], [top - 0.3, top - 0.3], color="black")

    for sec in sections:
        ax.text(COL_SUB, ypos[(sec, "__header__")], sec, fontsize=12, fontweight="bold", va="center")

    for row in rows:
        y = ypos[(row["section"], row["label"])]
        ax.text(COL_SUB + 0.04, y, row["label"], va="center")
        ax.text(COL_N, y, str(row["n"]), va="center")

        if row["is_reference"]:
            ax.text(COL_EST, y, "Reference", va="center")
            ax.scatter(map_x(1.0), y, s=50, marker="s", color="black") # Reference marker at OR=1
            ax.text(COL_P, y, "Ref.", va="center")
        else:
            d = row["diff"]
            lo = row["lo"]
            hi = row["hi"]

            # Removed the '+' sign format since ORs are naturally positive
            ax.text(COL_EST, y, f"{d:.2f} ({lo:.2f}, {hi:.2f})", va="center")
            ax.plot([map_x(lo), map_x(hi)], [y, y], lw=2, color="black")
            ax.scatter(map_x(d), y, s=50, marker="s", color="black")

            ptxt = "<0.001" if row["p"] < 0.001 else f"{row['p']:.3f}"
            ax.text(COL_P, y, ptxt, va="center")

    ax.plot([map_x(1.0), map_x(1.0)], [0, top], ls="--", color="gray") 
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
# STANDARD LOGISTIC REGRESSION RESULTS
# ==========================================================
def print_logit_results(df, outcome_col, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    reg_df = df.copy()
    reg_df = reg_df[["nlp_gender", "canonical_subtype", outcome_col, "age_group"]].copy()

    reg_df[outcome_col] = pd.to_numeric(reg_df[outcome_col], errors="coerce")
    reg_df = reg_df.dropna(subset=[outcome_col, "age_group"])
    reg_df = reg_df[np.isfinite(reg_df[outcome_col])].copy()

    print(f"\nBinary Outcome Summary ({outcome_col}):")
    print(reg_df[outcome_col].value_counts())

    reg_df["nlp_gender_M"] = (reg_df["nlp_gender"] == "M").astype(float)
    reg_df["canonical_subtype_General"] = (reg_df["canonical_subtype"] == "General").astype(float)
    reg_df["age_group_40-64"] = (reg_df["age_group"] == "40-64").astype(float)
    reg_df["age_group_65+"] = (reg_df["age_group"] == "65+").astype(float)

    X = reg_df[["nlp_gender_M", "canonical_subtype_General", "age_group_40-64", "age_group_65+"]].astype(float)
    X = sm.add_constant(X)
    
    y = reg_df[outcome_col].astype(float)

    # Fit Logistic Regression
    model = sm.Logit(y, X).fit()

    print(model.summary())
    print("\n95% CI (Log-Odds)")
    print(model.conf_int())
    
    print("\nOdds Ratios (Exponentiated Coefficients)")
    print(np.exp(model.params))

    return model

# ==========================================================
# LIKELIHOOD RATIO TESTS: M0 vs M1, M2, M3, M4, M5
# ==========================================================
def build_interaction_models(df, outcome_col):
    """Constructs dummy variables and interaction columns for M0-M5."""
    reg_df = df[["nlp_gender", "canonical_subtype", outcome_col, "age_group"]].copy()
    reg_df[outcome_col] = pd.to_numeric(reg_df[outcome_col], errors="coerce")
    reg_df = reg_df.dropna(subset=[outcome_col, "age_group", "nlp_gender", "canonical_subtype"])
    reg_df = reg_df[np.isfinite(reg_df[outcome_col])].copy()

    # Main effects
    reg_df["sex_M"] = (reg_df["nlp_gender"] == "M").astype(float)
    reg_df["type_Gen"] = (reg_df["canonical_subtype"] == "General").astype(float)
    reg_df["age_40_64"] = (reg_df["age_group"] == "40-64").astype(float)
    reg_df["age_65_plus"] = (reg_df["age_group"] == "65+").astype(float)

    # Two-way interactions
    # 1. sex:epilepsy_type
    reg_df["int_sex_type"] = reg_df["sex_M"] * reg_df["type_Gen"]
    
    # 2. epilepsy_type:age 
    reg_df["int_type_age40"] = reg_df["type_Gen"] * reg_df["age_40_64"]
    reg_df["int_type_age65"] = reg_df["type_Gen"] * reg_df["age_65_plus"]

    # 3. sex:age
    reg_df["int_sex_age40"] = reg_df["sex_M"] * reg_df["age_40_64"]
    reg_df["int_sex_age65"] = reg_df["sex_M"] * reg_df["age_65_plus"]

    # Three-way interaction: sex:epilepsy_type:age
    reg_df["int_3way_age40"] = reg_df["sex_M"] * reg_df["type_Gen"] * reg_df["age_40_64"]
    reg_df["int_3way_age65"] = reg_df["sex_M"] * reg_df["type_Gen"] * reg_df["age_65_plus"]

    # Define predictor sets for each model
    base_vars = ["sex_M", "type_Gen", "age_40_64", "age_65_plus"]
    
    models_dict = {
        "M0 (Base)": base_vars,
        "M1 (+ sex:type)": base_vars + ["int_sex_type"],
        "M2 (+ type:age)": base_vars + ["int_type_age40", "int_type_age65"],
        "M3 (+ sex:age)": base_vars + ["int_sex_age40", "int_sex_age65"],
        "M4 (All 2-way)": base_vars + [
            "int_sex_type", "int_type_age40", "int_type_age65", "int_sex_age40", "int_sex_age65"
        ],
        "M5 (+ 3-way)": base_vars + [
            "int_sex_type", "int_type_age40", "int_type_age65", "int_sex_age40", "int_sex_age65",
            "int_3way_age40", "int_3way_age65"
        ]
    }

    return reg_df, models_dict

def run_all_lrts(df, outcome_col, outcome_label):
    print("\n" + "#" * 125)
    print(f"LIKELIHOOD RATIO TEST SUITE & COMPREHENSIVE OR TABLE — {outcome_label}")
    print("#" * 125)

    reg_df, models_dict = build_interaction_models(df, outcome_col)
    y = reg_df[outcome_col].astype(float)

    X0 = sm.add_constant(reg_df[models_dict["M0 (Base)"]].astype(float))
    res0 = sm.Logit(y, X0).fit(disp=0)

    print(f"\nCohort Size (N): {len(reg_df):,}")
    print(f"M0 (Base Model) Log-Likelihood: {res0.llf:.4f} | AIC: {res0.aic:.4f} | BIC: {res0.bic:.4f} | Parameters: {len(res0.params)}")
    print("-" * 125)

    summary_rows = []
    detailed_rows = []

    for model_name, var_list in models_dict.items():
        X_mod = sm.add_constant(reg_df[var_list].astype(float))
        
        try:
            res_mod = sm.Logit(y, X_mod).fit(disp=0)
            
            if model_name == "M0 (Base)":
                lr_stat_str = "Base"
                df_diff_str = "Base"
                p_val_str = "Base"
                is_better = "Base"
                lr_stat = np.nan
                p_val_lrt = np.nan
            else:
                lr_stat = 2.0 * (res_mod.llf - res0.llf)
                lr_stat = max(0.0, lr_stat)
                df_diff = len(res_mod.params) - len(res0.params)
                p_val_lrt = stats.chi2.sf(lr_stat, df_diff)
                
                is_better = "YES (p<0.05)" if p_val_lrt < 0.05 else "No"
                lr_stat_str = f"{lr_stat:.4f}"
                df_diff_str = str(int(df_diff))
                p_val_str = f"{p_val_lrt:.4f}" if p_val_lrt >= 0.0001 else "<0.0001"
                
            summary_rows.append({
                "Model": model_name,
                "Log-Likelihood": f"{res_mod.llf:.4f}",
                "AIC": f"{res_mod.aic:.4f}",
                "BIC": f"{res_mod.bic:.4f}",
                "LR Stat (X2)": lr_stat_str,
                "df Diff": df_diff_str,
                "LRT P-Value": p_val_str,
                "Better than M0?": is_better
            })

            conf_int = res_mod.conf_int()
            
            for param in res_mod.params.index:
                est = res_mod.params[param]
                ci_low = conf_int[0][param]
                ci_hi = conf_int[1][param]
                pval = res_mod.pvalues[param]
                
                or_est = np.exp(est)
                or_ci_low = np.exp(ci_low)
                or_ci_hi = np.exp(ci_hi)
                
                detailed_rows.append({
                    "Model": model_name,
                    "Term": param,
                    "Odds Ratio": f"{or_est:.4f}",
                    "95% CI Lower (OR)": f"{or_ci_low:.4f}",
                    "95% CI Upper (OR)": f"{or_ci_hi:.4f}",
                    "P-Value": f"{pval:.4f}" if pval >= 0.0001 else "<0.0001",
                    "Model LRT Stat": f"{lr_stat:.4f}" if not np.isnan(lr_stat) else "Base",
                    "Model LRT p-val": f"{p_val_lrt:.4f}" if not np.isnan(p_val_lrt) else "Base"
                })

        except np.linalg.LinAlgError:
            summary_rows.append({
                "Model": model_name, "Log-Likelihood": "N/A", "AIC": "N/A", "BIC": "N/A",
                "LR Stat (X2)": "N/A", "df Diff": "N/A", "LRT P-Value": "N/A", "Better than M0?": "Failed"
            })
        except Exception:
            pass

    print("\n" + "=" * 125)
    print(f"DETAILED ODDS RATIO TABLE: M0-M5 ({outcome_label})")
    print("=" * 125)
    detailed_df = pd.DataFrame(detailed_rows)
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000, 'display.colheader_justify', 'left'):
        print(detailed_df.to_string(index=False))

    print("\n" + "=" * 125)
    print(f"SUMMARY TABLE: MODEL FIT & LIKELIHOOD RATIO TESTS vs. M0 BASE ({outcome_label})")
    print("=" * 125)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
    print("=" * 125 + "\n")

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

    # ------------------------------------------------------------------
    # DIAGNOSTIC CHECK 1: Identify & Remove Missing Age (NaN)
    # ------------------------------------------------------------------
    before_pats = current_patients.copy()
    missing_age_mask = patient_df['mean_age_spike'].isna() | patient_df['median_age_sz'].isna()
    missing_age_patients = patient_df[missing_age_mask]['Patient'].tolist()
    
    print("-" * 60)
    if len(missing_age_patients) > 0:
        print(f"-> [DIAGNOSTIC] Found {len(missing_age_patients)} patient(s) with MISSING AGE (NaN):")
        print(f"   Patient IDs: {missing_age_patients}")
    else:
        print("-> [DIAGNOSTIC] No patients found with NaN age.")
    print("-" * 60 + "\n")
        
    patient_df = patient_df[~missing_age_mask]
    current_patients = set(patient_df['Patient'].unique())
    track_patients("Project Filter C1: Remove Missing Age (NaN)", before_pats, current_patients)

    # ------------------------------------------------------------------
    # DIAGNOSTIC CHECK 2: Identify & Remove Pediatric / Invalid (< 18.0)
    # ------------------------------------------------------------------
    before_pats = current_patients.copy()
    invalid_age_mask = (patient_df['mean_age_spike'] < 18.0) | (patient_df['median_age_sz'] < 18.0)
    invalid_age_patients = patient_df[invalid_age_mask]['Patient'].tolist()
    
    print("-" * 60)
    if len(invalid_age_patients) > 0:
        print(f"-> [DIAGNOSTIC] Found {len(invalid_age_patients)} patient(s) with AGE < 18 YEARS (Pediatric/Invalid):")
        for pid in invalid_age_patients:
            row = patient_df[patient_df['Patient'] == pid].iloc[0]
            print(f"   Patient ID: {pid} | mean_age_spike: {row['mean_age_spike']:.2f} | median_age_sz: {row['median_age_sz']:.2f}")
    else:
        print("-> [DIAGNOSTIC] No patients found with age < 18 years.")
    print("-" * 60 + "\n")
        
    patient_df = patient_df[~invalid_age_mask]
    current_patients = set(patient_df['Patient'].unique())
    track_patients("Project Filter C2: Keep Valid Adult Age (>= 18 years)", before_pats, current_patients)

    # --------------------------------------------------------
    # CREATE CATEGORICAL AGE GROUP (18-39 as reference)
    # --------------------------------------------------------
    bins = [18, 40, 65, np.inf] # [18, 40) = 18-39.99
    labels = ['18-39', '40-64', '65+']
    patient_df['age_group'] = pd.cut(patient_df['mean_age_spike'], bins=bins, labels=labels, right=False)
    patient_df['age_group'] = pd.Categorical(patient_df['age_group'], categories=labels, ordered=True)

    # --- BINARY CONVERSION BASED ON DYNAMIC MEDIANS ---
    med_spike = patient_df["spike_rate_per_hour"].median()
    med_sz = patient_df["mean_sz_freq"].median()
    
    patient_df["spike_rate_binary"] = (patient_df["spike_rate_per_hour"] >= med_spike).astype(int)
    patient_df["sz_freq_binary"] = (patient_df["mean_sz_freq"] >= med_sz).astype(int)

    print("=" * 75)
    print(f"FINAL PATIENT COHORT SIZE: {len(current_patients):,} patients")
    print(f"  -> Median Spike Rate Split:     {med_spike:.2f} / hr")
    print(f"  -> Median Seizure Freq Split:   {med_sz:.2f} / mo")
    print("=" * 75)

    print("\n======================================================")
    print("               AGE GROUP DISTRIBUTION                 ")
    print("======================================================")
    print(patient_df['age_group'].value_counts().sort_index())
    print("======================================================\n")

    # 6. STANDARD LOGISTIC REGRESSION RESULTS (M0 Base)
    print_logit_results(patient_df, "spike_rate_binary", f"LOGISTIC REGRESSION: SPIKE RATE (≥{med_spike:.2f}/hr)")
    print_logit_results(patient_df, "sz_freq_binary", f"LOGISTIC REGRESSION: SEIZURE FREQUENCY (≥{med_sz:.2f}/mo)")

    # 7. COMPREHENSIVE LIKELIHOOD RATIO TEST SUITE (M1 - M5 vs M0 Base)
    run_all_lrts(patient_df, "spike_rate_binary", f"SPIKE RATE (≥{med_spike:.2f}/hr)")
    run_all_lrts(patient_df, "sz_freq_binary", f"SEIZURE FREQUENCY (≥{med_sz:.2f}/mo)")

    # # 8. RUN BOOTSTRAP ANALYSIS
    # rows_spike = run_analysis(patient_df, "spike_rate_binary", f"SPIKE RATE (≥{med_spike:.2f}/hr)")
    # rows_sz    = run_analysis(patient_df, "sz_freq_binary", f"SEIZURE FREQUENCY (≥{med_sz:.2f}/mo)")

    # # 9. GENERATE FOREST PLOTS
    # print("\n--- Generating Forest Plots ---")
    # forest_plot(
    #     rows_spike,
    #     f"Bootstrapped Logit Coefficients: Spike Rate (≥{med_spike:.2f}/hr) ~ Sex + Type + Age Group",
    #     (-0.4, 0.8),
    #     [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
    #     "Coefficient (Log-Odds)", 
    #     os.path.join(save_dir, "forest_spike_rate_boot_binary_agegroup_dynamic.png")
    # )

    # forest_plot(
    #     rows_sz,
    #     f"Bootstrapped Logit Coefficients: Seizure Freq (≥{med_sz:.2f}/mo) ~ Sex + Type + Age Group",
    #     (-1.2, 0.2),
    #     [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2],
    #     "Coefficient (Log-Odds)", 
    #     os.path.join(save_dir, "forest_seizure_freq_boot_binary_agegroup_dynamic.png")
    # )
    # print("Done plotting. All outputs saved to:", save_dir)

if __name__ == "__main__":
    main()