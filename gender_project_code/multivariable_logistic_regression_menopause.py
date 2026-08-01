# Bootstrapped binary forest plot of Logit models for spike rate and seizure frequency 
# Modeled by sex and pre-menopausal vs. post-menopausal age (<51 vs >=51) 
# with median split for binary spike rate and seizure frequency outcomes

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
# TRACKING, PARSING & SUBTYPE HELPERS
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
    if etype == "general" or etype == "generalized":
        return "Generalized"
    return np.nan

# ==========================================================
# COHORT STATS GENERATOR & IMAGE/TABLE SAVER
# ==========================================================
def print_and_save_requested_stats(patient_df, valid_sessions, vuniq, save_dir, n_boot=2000):
    print("\n" + "=" * 75)
    print("FINAL COHORT SUMMARY STATISTICS (TABLE 1)")
    print("=" * 75)
    
    cohort_pids = set(patient_df['Patient'].unique())
    n_patients = len(cohort_pids)
    
    cohort_sessions = valid_sessions[valid_sessions['Patient'].isin(cohort_pids)].copy()
    cohort_visits = vuniq[vuniq['Patient'].isin(cohort_pids)].copy()

    def median_iqr(series, decimals=1):
        s = series.dropna()
        if len(s) == 0: return "N/A"
        med = s.median()
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        return f"{med:.{decimals}f} ({q25:.{decimals}f}-{q75:.{decimals}f})"

    def boot_median_ci(series, decimals=2, n_boot=2000):
        s = series.dropna().values
        if len(s) == 0: return "N/A"
        medians = [np.median(np.random.choice(s, size=len(s), replace=True)) for _ in range(n_boot)]
        lo, hi = np.percentile(medians, 2.5), np.percentile(medians, 97.5)
        return f"[{lo:.{decimals}f} - {hi:.{decimals}f}]"

    n_males = len(patient_df[patient_df['nlp_gender'] == 'M'])
    n_females = len(patient_df[patient_df['nlp_gender'] == 'F'])
    n_focal = len(patient_df[patient_df['canonical_subtype'] == 'Focal'])
    n_generalized = len(patient_df[patient_df['canonical_subtype'] == 'Generalized'])
    n_18_39 = len(patient_df[patient_df['age_group'] == '18-39'])
    n_40_64 = len(patient_df[patient_df['age_group'] == '40-64'])
    n_65_plus = len(patient_df[patient_df['age_group'] == '65+'])

    visits_per_pat = cohort_visits.groupby('Patient').size()
    cohort_visits['VisitDate'] = pd.to_datetime(cohort_visits['VisitDate'], errors='coerce')
    fup = cohort_visits.groupby('Patient')['VisitDate'].agg(lambda x: (x.max() - x.min()).days / 365.25)
    
    doc_pct = cohort_visits.groupby('Patient')['had_doc_freq'].mean() * 100
    eegs_per_pat = cohort_sessions.groupby('Patient').size()

    sz_s = patient_df['mean_sz_freq']
    spk_s = patient_df['spike_rate_per_hour']

    eeg_spikes_present = len(cohort_sessions[cohort_sessions['count_0_46'] > 0])
    eeg_total = len(cohort_sessions)
    eeg_pct = (eeg_spikes_present / eeg_total) * 100 if eeg_total > 0 else 0

    table_data = [
        {"Category": "Total Patients", "Metric": "N", "Value": f"{n_patients:,}", "Bootstrapped 95% CI": ""},
        {"Category": "Sex", "Metric": "Men N (%)", "Value": f"{n_males:,} ({(n_males/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "Sex", "Metric": "Women N (%)", "Value": f"{n_females:,} ({(n_females/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "Epilepsy Subtype", "Metric": "Focal Lobe N (%)", "Value": f"{n_focal:,} ({(n_focal/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "Epilepsy Subtype", "Metric": "Generalized N (%)", "Value": f"{n_generalized:,} ({(n_generalized/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "Age Group", "Metric": "18-39 years N (%)", "Value": f"{n_18_39:,} ({(n_18_39/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "Age Group", "Metric": "40-64 years N (%)", "Value": f"{n_40_64:,} ({(n_40_64/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "Age Group", "Metric": "65+ years N (%)", "Value": f"{n_65_plus:,} ({(n_65_plus/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "Clinical Follow-up", "Metric": "Number of clinic visits Median (IQR)", "Value": median_iqr(visits_per_pat, 1), "Bootstrapped 95% CI": ""},
        {"Category": "Clinical Follow-up", "Metric": "Years of follow-up Median (IQR)", "Value": median_iqr(fup, 1), "Bootstrapped 95% CI": ""},
        {"Category": "Clinical Follow-up", "Metric": "% Visits with documented sz freq Median (IQR)", "Value": f"{median_iqr(doc_pct, 1)}%", "Bootstrapped 95% CI": ""},
        {"Category": "EEG Metrics", "Metric": "Number of EEGs Median (IQR)", "Value": median_iqr(eegs_per_pat, 1), "Bootstrapped 95% CI": ""},
        {"Category": "Seizure Frequency", "Metric": "Mean monthly seizures across visits Median (IQR)", "Value": median_iqr(sz_s, 2), "Bootstrapped 95% CI": boot_median_ci(sz_s, 2)},
        {"Category": "Spike Rate", "Metric": "Mean spikes/hour across EEGs Median (IQR)", "Value": median_iqr(spk_s, 2), "Bootstrapped 95% CI": boot_median_ci(spk_s, 2)},
        {"Category": "EEG Metrics", "Metric": "EEGs with reported spikes - Present N (%)", "Value": f"{eeg_spikes_present:,} ({eeg_pct:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "EEG Metrics", "Metric": "EEGs with reported spikes - Absent N (%)", "Value": f"{eeg_total - eeg_spikes_present:,} ({100 - eeg_pct:.1f}%)", "Bootstrapped 95% CI": ""}
    ]

    summary_df = pd.DataFrame(table_data)
    print(summary_df.to_string(index=False))
    print("=" * 75 + "\n")

    csv_path = os.path.join(save_dir, "final_cohort_table1_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    
    html_path = os.path.join(save_dir, "final_cohort_table1_summary.html")
    with open(html_path, "w") as f:
        f.write("<html><head><title>Table 1 Summary</title></head><body>")
        f.write("<h2>Final Cohort Summary Statistics (Table 1)</h2>")
        f.write(summary_df.to_html(index=False))
        f.write("</body></html>")

    png_path = os.path.join(save_dir, "final_cohort_table1_summary.png")
    fig, ax = plt.subplots(figsize=(15, len(summary_df) * 0.40 + 1.2))
    ax.axis("off")
    
    table = ax.table(
        cellText=summary_df.values.tolist(),
        colLabels=summary_df.columns.tolist(),
        colWidths=[0.18, 0.46, 0.18, 0.18],
        loc="upper center",
        cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row_idx == 0:
            cell.set_facecolor("#000000")
            cell.set_text_props(color="white", fontweight="bold")
            if col_idx in [2, 3]: cell.set_text_props(ha="right")
        else:
            cell.set_facecolor("#F8FAFC" if row_idx % 2 == 0 else "white")
            if col_idx in [2, 3]: cell.set_text_props(ha="right")
                
    ax.set_title("Final Cohort Baseline Characteristics (Table 1)", fontweight="bold", fontsize=16, y=0.98, pad=10)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"-> Saved Table 1 outputs to:\n   1. {png_path}\n   2. {csv_path}\n")

# ==========================================================
# SUPPLEMENTARY TABLE S1: MENOPAUSAL AGE STRATIFICATION
# ==========================================================
def print_and_save_supplementary_table(patient_df, save_dir):
    print("\n" + "=" * 80)
    print("SUPPLEMENTARY TABLE S1: COHORT BY MENOPAUSAL AGE STRATUM (<51 vs >=51)")
    print("=" * 80)

    pre_df = patient_df[patient_df['mean_age_spike'] < 51.0]
    post_df = patient_df[patient_df['mean_age_spike'] >= 51.0]

    def get_stats(df):
        n = len(df)
        if n == 0: return {}
        males = len(df[df['nlp_gender'] == 'M'])
        females = len(df[df['nlp_gender'] == 'F'])
        focal = len(df[df['canonical_subtype'] == 'Focal'])
        generalized = len(df[df['canonical_subtype'] == 'Generalized'])
        med_spk, q25_spk, q75_spk = df['spike_rate_per_hour'].median(), df['spike_rate_per_hour'].quantile(0.25), df['spike_rate_per_hour'].quantile(0.75)
        med_sz, q25_sz, q75_sz = df['mean_sz_freq'].median(), df['mean_sz_freq'].quantile(0.25), df['mean_sz_freq'].quantile(0.75)
        return {
            "N": f"{n:,}",
            "Women N (%)": f"{females:,} ({(females/n)*100:.1f}%)",
            "Men N (%)": f"{males:,} ({(males/n)*100:.1f}%)",
            "Focal Subtype N (%)": f"{focal:,} ({(focal/n)*100:.1f}%)",
            "Generalized Subtype N (%)": f"{generalized:,} ({(generalized/n)*100:.1f}%)",
            "Spike Rate Median (IQR)": f"{med_spk:.2f} ({q25_spk:.2f}-{q75_spk:.2f})",
            "Seizure Freq Median (IQR)": f"{med_sz:.2f} ({q25_sz:.2f}-{q75_sz:.2f})"
        }

    stats_pre = get_stats(pre_df)
    stats_post = get_stats(post_df)

    metrics = [
        "N", "Women N (%)", "Men N (%)", 
        "Focal Subtype N (%)", "Generalized Subtype N (%)", 
        "Spike Rate Median (IQR)", "Seizure Freq Median (IQR)"
    ]

    table_data = []
    for m in metrics:
        table_data.append({
            "Characteristic / Outcome": m,
            "Pre-Menopausal Age (< 51 years)": stats_pre.get(m, "N/A"),
            "Post-Menopausal Age (≥ 51 years)": stats_post.get(m, "N/A")
        })

    supp_df = pd.DataFrame(table_data)
    print(supp_df.to_string(index=False))
    print("=" * 80 + "\n")

    csv_path = os.path.join(save_dir, "final_cohort_tableS1_menopause_stratified.csv")
    supp_df.to_csv(csv_path, index=False)

    png_path = os.path.join(save_dir, "final_cohort_tableS1_menopause_stratified.png")
    fig, ax = plt.subplots(figsize=(14, len(supp_df) * 0.45 + 1.2))
    ax.axis("off")
    
    table = ax.table(
        cellText=supp_df.values.tolist(),
        colLabels=supp_df.columns.tolist(),
        colWidths=[0.38, 0.31, 0.31],
        loc="upper center",
        cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row_idx == 0:
            cell.set_facecolor("#000000")
            cell.set_text_props(color="white", fontweight="bold")
            if col_idx in [1, 2]: cell.set_text_props(ha="right")
        else:
            cell.set_facecolor("#F8FAFC" if row_idx % 2 == 0 else "white")
            if col_idx in [1, 2]: cell.set_text_props(ha="right")
                
    ax.set_title("Table S1: Cohort Characteristics Stratified by Menopausal Age Cutoff (51 Years)", fontweight="bold", fontsize=15, y=0.98, pad=10)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"-> Saved Supplementary Table S1 outputs to:\n   1. {png_path}\n   2. {csv_path}\n")

# ==========================================================
# LOGISTIC BOOTSTRAP HELPERS
# ==========================================================
def bootstrap_regression_coeffs(df, outcome_col, formula_vars, n_boot=5000):
    df = df.copy()
    df = df[np.isfinite(df[outcome_col])]

    cols_to_dummy = [outcome_col, "nlp_gender", "canonical_subtype"]
    if "age_group" in df.columns and any("age_group" in v for v in formula_vars):
        cols_to_dummy.append("age_group")

    reg_df = pd.get_dummies(df[cols_to_dummy], drop_first=True).astype(float)
    results = {}

    for var in formula_vars:
        if var not in reg_df.columns:
            results[var] = (1, 1, 1, 1)
            continue

        coeffs = []
        for _ in range(n_boot):
            sample = reg_df.sample(len(reg_df), replace=True)
            X = sm.add_constant(sample[formula_vars])
            y = sample[outcome_col]

            try:
                model = sm.Logit(y, X).fit(disp=0)
                beta = model.params[var]
                if np.isfinite(beta): coeffs.append(beta)
            except Exception:
                pass

        coeffs = np.asarray(coeffs)
        if len(coeffs) == 0:
            results[var] = (1, 1, 1, 1)
            continue

        p = 2 * min(np.mean(coeffs >= 0), np.mean(coeffs <= 0))
        results[var] = (
            float(np.exp(np.mean(coeffs))),
            float(np.exp(np.percentile(coeffs, 2.5))),
            float(np.exp(np.percentile(coeffs, 97.5))),
            float(p)
        )
    return results

def run_analysis(patient_df, outcome_col, outcome_label):
    print(f"\nBOOTSTRAP LOGIT REGRESSION (PRIMARY) — {outcome_label}")
    formula_vars = ['nlp_gender_M', 'canonical_subtype_Generalized', 'age_group_40-64', 'age_group_65+']
    coeffs = bootstrap_regression_coeffs(patient_df, outcome_col, formula_vars)
    
    rows = []
    m_res = coeffs['nlp_gender_M']
    rows.append(dict(section="Sex", label="Female", n=len(patient_df[patient_df['nlp_gender']=='F']), is_reference=True, diff=1, lo=1, hi=1, p=None))
    rows.append(dict(section="Sex", label="Male", n=len(patient_df[patient_df['nlp_gender']=='M']), is_reference=False, diff=m_res[0], lo=m_res[1], hi=m_res[2], p=m_res[3]))

    g_res = coeffs['canonical_subtype_Generalized']
    rows.append(dict(section="Epilepsy Type", label="Focal", n=len(patient_df[patient_df['canonical_subtype']=='Focal']), is_reference=True, diff=1, lo=1, hi=1, p=None))
    rows.append(dict(section="Epilepsy Type", label="Generalized", n=len(patient_df[patient_df['canonical_subtype']=='Generalized']), is_reference=False, diff=g_res[0], lo=g_res[1], hi=g_res[2], p=g_res[3]))

    rows.append(dict(section="Age Group", label="18-39 years", n=len(patient_df[patient_df['age_group']=='18-39']), is_reference=True, diff=1, lo=1, hi=1, p=None))
    a2_res = coeffs['age_group_40-64']
    rows.append(dict(section="Age Group", label="40-64 years", n=len(patient_df[patient_df['age_group']=='40-64']), is_reference=False, diff=a2_res[0], lo=a2_res[1], hi=a2_res[2], p=a2_res[3]))
    a3_res = coeffs['age_group_65+']
    rows.append(dict(section="Age Group", label="65+ years", n=len(patient_df[patient_df['age_group']=='65+']), is_reference=False, diff=a3_res[0], lo=a3_res[1], hi=a3_res[2], p=a3_res[3]))

    return rows

# ==========================================================
# SUPPLEMENTARY ANALYSIS: MENOPAUSAL AGE STRATIFIED MODEL
# ==========================================================
def run_menopausal_stratified_analysis(patient_df, outcome_col, outcome_label):
    print(f"\nBOOTSTRAP LOGIT REGRESSION (MENOPAUSE STRATIFIED) — {outcome_label}")
    
    pre_df = patient_df[patient_df['mean_age_spike'] < 51.0].copy()
    post_df = patient_df[patient_df['mean_age_spike'] >= 51.0].copy()
    
    formula_vars = ['nlp_gender_M', 'canonical_subtype_Generalized']
    
    print(f"  -> Running Pre-Menopausal Cohort (<51 yrs, N={len(pre_df)})...")
    coeffs_pre = bootstrap_regression_coeffs(pre_df, outcome_col, formula_vars)
    
    print(f"  -> Running Post-Menopausal Cohort (>=51 yrs, N={len(post_df)})...")
    coeffs_post = bootstrap_regression_coeffs(post_df, outcome_col, formula_vars)
    
    rows = []
    
    m_pre = coeffs_pre['nlp_gender_M']
    g_pre = coeffs_pre['canonical_subtype_Generalized']
    
    m_post = coeffs_post['nlp_gender_M']
    g_post = coeffs_post['canonical_subtype_Generalized']
    
    # 1. Sex: Pre-Menopausal
    rows.append(dict(section="Sex: Pre-Menopause (< 51 yrs)", label="Female", n=len(pre_df[pre_df['nlp_gender']=='F']), is_reference=True, diff=1, lo=1, hi=1, p=None))
    rows.append(dict(section="Sex: Pre-Menopause (< 51 yrs)", label="Male", n=len(pre_df[pre_df['nlp_gender']=='M']), is_reference=False, diff=m_pre[0], lo=m_pre[1], hi=m_pre[2], p=m_pre[3]))

    # 2. Sex: Post-Menopausal
    rows.append(dict(section="Sex: Post-Menopause (≥ 51 yrs)", label="Female", n=len(post_df[post_df['nlp_gender']=='F']), is_reference=True, diff=1, lo=1, hi=1, p=None))
    rows.append(dict(section="Sex: Post-Menopause (≥ 51 yrs)", label="Male", n=len(post_df[post_df['nlp_gender']=='M']), is_reference=False, diff=m_post[0], lo=m_post[1], hi=m_post[2], p=m_post[3]))
    
    # 3. Type: Pre-Menopausal
    rows.append(dict(section="Epilepsy Type: Pre-Menopause (< 51 yrs)", label="Focal", n=len(pre_df[pre_df['canonical_subtype']=='Focal']), is_reference=True, diff=1, lo=1, hi=1, p=None))
    rows.append(dict(section="Epilepsy Type: Pre-Menopause (< 51 yrs)", label="Generalized", n=len(pre_df[pre_df['canonical_subtype']=='Generalized']), is_reference=False, diff=g_pre[0], lo=g_pre[1], hi=g_pre[2], p=g_pre[3]))

    # 4. Type: Post-Menopausal
    rows.append(dict(section="Epilepsy Type: Post-Menopause (≥ 51 yrs)", label="Focal", n=len(post_df[post_df['canonical_subtype']=='Focal']), is_reference=True, diff=1, lo=1, hi=1, p=None))
    rows.append(dict(section="Epilepsy Type: Post-Menopause (≥ 51 yrs)", label="Generalized", n=len(post_df[post_df['canonical_subtype']=='Generalized']), is_reference=False, diff=g_post[0], lo=g_post[1], hi=g_post[2], p=g_post[3]))

    return rows

# ==========================================================
# FEMALE ONLY: DESCRIPTIVES & PRE- VS POST-MENOPAUSE LOGIT
# ==========================================================
def compare_females_menopause(patient_df):
    print("\n" + "=" * 80)
    print("FEMALE COHORT: PRE- VS POST-MENOPAUSE DETAILED PROFILE (<51 vs >=51)")
    print("=" * 80)
    
    f_df = patient_df[patient_df['nlp_gender'] == 'F'].copy()
    f_df['post_menopause'] = (f_df['mean_age_spike'] >= 51.0).astype(float)
    f_df['canonical_subtype_Generalized'] = (f_df['canonical_subtype'] == 'Generalized').astype(float)
    
    pre_f = f_df[f_df['post_menopause'] == 0.0]
    post_f = f_df[f_df['post_menopause'] == 1.0]
    
    def get_iqr_string(series):
        return f"{series.median():.2f} ({series.quantile(0.25):.2f} - {series.quantile(0.75):.2f})"
        
    print(f"DESCRIPTIVE STATISTICS SUMMARY:")
    print(f"  Total Female Patients Evaluated:  {len(f_df)}")
    print(f"    - Pre-Menopausal Age (<51):    {len(pre_f)} patients")
    print(f"    - Post-Menopausal Age (>=51):  {len(post_f)} patients")
    print("\n  Epilepsy Subtype Distribution:")
    print(f"    - Pre-Menopause Focal:         {len(pre_f[pre_f['canonical_subtype']=='Focal'])} ({(len(pre_f[pre_f['canonical_subtype']=='Focal'])/len(pre_f))*100:.1f}%)")
    print(f"    - Pre-Menopause Generalized:   {len(pre_f[pre_f['canonical_subtype']=='Generalized'])} ({(len(pre_f[pre_f['canonical_subtype']=='Generalized'])/len(pre_f))*100:.1f}%)")
    print(f"    - Post-Menopause Focal:        {len(post_f[post_f['canonical_subtype']=='Focal'])} ({(len(post_f[post_f['canonical_subtype']=='Focal'])/len(post_f))*100:.1f}%)")
    print(f"    - Post-Menopause Generalized:  {len(post_f[post_f['canonical_subtype']=='Generalized'])} ({(len(post_f[post_f['canonical_subtype']=='Generalized'])/len(post_f))*100:.1f}%)")
    print("\n  Outcome Values Median (IQR):")
    print(f"    - Spike Rate/Hr (Pre):         {get_iqr_string(pre_f['spike_rate_per_hour'])}")
    print(f"    - Spike Rate/Hr (Post):        {get_iqr_string(post_f['spike_rate_per_hour'])}")
    print(f"    - Seizure Freq/Mo (Pre):       {get_iqr_string(pre_f['mean_sz_freq'])}")
    print(f"    - Seizure Freq/Mo (Post):      {get_iqr_string(post_f['mean_sz_freq'])}")
    print("-" * 80)

    for outcome in ["spike_rate_binary", "sz_freq_binary"]:
        print(f"\nREGRESSION ANALYSIS — Outcome: {outcome.replace('_', ' ').upper()}")
        tmp = f_df[['post_menopause', 'canonical_subtype_Generalized', outcome]].dropna()
        tmp[outcome] = pd.to_numeric(tmp[outcome], errors="coerce")
        tmp = tmp[np.isfinite(tmp[outcome])]

        X = sm.add_constant(tmp[['post_menopause', 'canonical_subtype_Generalized']])
        y = tmp[outcome].astype(float)
        
        try:
            model = sm.Logit(y, X).fit(disp=0)
            print(model.summary())
            print("\nOdds Ratios (Exponentiated Coefficients):")
            print(np.exp(model.params))
            
            p_val = model.pvalues['post_menopause']
            if p_val < 0.05:
                print(f"\n=> SIGNIFICANT difference found between pre- and post-menopausal females (p = {p_val:.4f}).")
            else:
                print(f"\n=> NO significant difference found between pre- and post-menopausal females (p = {p_val:.4f}).")
        except Exception as e:
            print(f"Model fitting failed for {outcome}: {e}")

# ==========================================================
# CUSTOM FOREST PLOT (WITH UNIFORM SPACING & STYLING)
# ==========================================================
def forest_plot(rows, title, x_lim, x_ticks, x_label, left_dir_label, right_dir_label, out_path):
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.axis("off")

    sections = []
    for r in rows:
        if r["section"] not in sections:
            sections.append(r["section"])

    COL_SUB = 0.02
    COL_N = 0.22 
    COL_EST = 0.38
    PLOT_L = 0.52   
    PLOT_R = 0.90
    COL_P = 0.93

    def map_x(v):
        frac = (v - x_lim[0]) / (x_lim[1] - x_lim[0])
        return PLOT_L + frac * (PLOT_R - PLOT_L)

    # --- MATH FOR PERFECT UNIFORM SPACING ---
    top = 10.0
    section_gap = 0.85  # Clean whitespace between sections
    line_gap = 0.65     # Tighter row grouping
    
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

    # Top row headers
    ax.text(COL_SUB, top, "Subgroup / Variable", fontweight="bold", va="center")
    ax.text(COL_N, top, "Patients (N)", fontweight="bold", va="center")
    ax.text(COL_EST, top, "Odds Ratio (95% CI)", fontweight="bold", va="center") 
    ax.text(COL_P, top, "P-Value", fontweight="bold", va="center")
    
    # Positioned above the graphical dashed line
    ax.text(map_x(1.0), top, "Odds Ratio (95% CI)", fontweight="bold", ha="center", va="center")

    # Top black dividing line
    ax.plot([0, 1], [top - 0.3, top - 0.3], color="black")

    for sec in sections:
        ax.text(COL_SUB, ypos[(sec, "__header__")], sec, fontsize=12, fontweight="bold", va="center")

    for row in rows:
        y_loc = ypos[(row["section"], row["label"])]
        
        # Dynamically append (Ref.) to reference categories
        label_text = f'{row["label"]} (Ref.)' if row["is_reference"] else row["label"]
        ax.text(COL_SUB + 0.04, y_loc, label_text, va="center")
        
        ax.text(COL_N, y_loc, str(row["n"]), va="center")

        if row["is_reference"]:
            ax.text(COL_EST, y_loc, "Ref.", va="center")
            ax.scatter(map_x(1.0), y_loc, s=50, marker="s", color="black")
            ax.text(COL_P, y_loc, "Ref.", va="center")
        else:
            d = row["diff"]
            lo = row["lo"]
            hi = row["hi"]

            ax.text(COL_EST, y_loc, f"{d:.2f} ({lo:.2f}-{hi:.2f})", va="center")
            ax.plot([map_x(lo), map_x(hi)], [y_loc, y_loc], lw=2, color="black")
            ax.scatter(map_x(d), y_loc, s=50, marker="s", color="black")

            ptxt = "<0.001" if row["p"] < 0.001 else f"{row['p']:.3f}"
            ax.text(COL_P, y_loc, ptxt, va="center")

    # Find the lowest row dynamically and snap the axis right below it
    bottom_y = min(ypos.values())
    axis_y = bottom_y - 0.8  
    
    # Reference dashed line locked to Odds Ratio = 1.0
    ax.plot([map_x(1.0), map_x(1.0)], [axis_y, top - 0.3], ls="--", color="gray")
    
    # Bottom black line
    ax.plot([map_x(x_lim[0]), map_x(x_lim[1])], [axis_y, axis_y], color="black")

    for tick in x_ticks:
        x_loc = map_x(tick)
        ax.plot([x_loc, x_loc], [axis_y, axis_y - 0.15], color="black") # Tick marks
        ax.text(x_loc, axis_y - 0.25, str(tick), ha="center", va="top") # Tick numbers

    # =========================================================================
    # DOUBLE-SIDED DIRECTIONAL ARROW (<--->)
    # =========================================================================
    arrow_y = axis_y - 1.0
    
    axis_range = x_lim[1] - x_lim[0]
    
    arrow_left = map_x(x_lim[0] + axis_range * 0.05)
    arrow_right = map_x(x_lim[0] + axis_range * 0.95)
    
    ax.annotate("", xy=(arrow_left, arrow_y), xytext=(arrow_right, arrow_y),
                arrowprops=dict(arrowstyle="<->", lw=1.8, color="black"))
    
    text_left_x = map_x(x_lim[0] + axis_range * 0.25)
    text_right_x = map_x(x_lim[0] + axis_range * 0.75)
    
    ax.text(text_left_x, arrow_y + 0.15, left_dir_label, 
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="black")
    ax.text(text_right_x, arrow_y + 0.15, right_dir_label, 
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="black")

    plt.title(title, fontweight="bold", pad=15)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# ==========================================================
# STANDARD LOGISTIC REGRESSION RESULTS
# ==========================================================
def print_logit_results(df, outcome_col, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    reg_df = df.copy()[["nlp_gender", "canonical_subtype", outcome_col, "age_group"]].dropna()
    reg_df[outcome_col] = pd.to_numeric(reg_df[outcome_col], errors="coerce")
    reg_df = reg_df[np.isfinite(reg_df[outcome_col])].copy()

    reg_df["nlp_gender_M"] = (reg_df["nlp_gender"] == "M").astype(float)
    reg_df["canonical_subtype_Generalized"] = (reg_df["canonical_subtype"] == "Generalized").astype(float)
    reg_df["age_group_40-64"] = (reg_df["age_group"] == "40-64").astype(float)
    reg_df["age_group_65+"] = (reg_df["age_group"] == "65+").astype(float)

    X = sm.add_constant(reg_df[["nlp_gender_M", "canonical_subtype_Generalized", "age_group_40-64", "age_group_65+"]].astype(float))
    y = reg_df[outcome_col].astype(float)
    model = sm.Logit(y, X).fit(disp=0)

    print(model.summary())
    print("\nOdds Ratios (Exponentiated Coefficients):")
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

    clinical_df['start_time_deid'] = pd.to_datetime(clinical_df['start_time_deid'], errors='coerce')
    clinical_df['deid_birth_date'] = pd.to_datetime(clinical_df['deid_birth_date'], errors='coerce')
    clinical_df['age'] = (clinical_df['start_time_deid'] - clinical_df['deid_birth_date']).dt.days / 365.25

    print("=" * 75)
    print("DATA CLEANING & COHORT ATTRITION")
    print("=" * 75)

    current_patients = set(clinical_df['Patient'].dropna().unique())
    before_pats = current_patients.copy()
    
    # 1. OUTPATIENT ROUTINE
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
    current_patients = {pid for pid in current_patients if str(pt_demo.loc[pid, 'epilepsy_type']).lower().strip() not in bad_types}
    track_patients("Base Filter: LLM-Confirmed Epilepsy Diagnosis", before_pats, current_patients)

    # 3. SEIZURE FREQUENCY
    before_pats = current_patients.copy()
    allowable_visits = {"CONSULT VISIT", "ESTABLISHED PATIENT VISIT", "FOLLOW-UP PATIENT CLINIC", "NEW PATIENT CLINIC", "NEW PATIENT VISIT", "NPV MANAGEMENT DURING COVID-19", "NPV NEUROLOGY", "RETURN ANNUAL VISIT", "RETURN PATIENT EXTENDED", "RETURN PATIENT VISIT", "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN"}
    
    flat_visits = []
    for _, row in clinical_df[clinical_df['Patient'].isin(current_patients)].iterrows():
        pid = row['Patient']
        v_dates, v_types = parse_json_array(row.get('visit_dates_deid')), parse_json_array(row.get('visit_type'))
        v_freqs, v_has_sz = parse_json_array(row.get('sz_freqs'), is_numeric=True), parse_json_array(row.get('visit_hasSz'), is_numeric=True, is_hassz=True)
        for i in range(len(v_dates)):
            if i < len(v_types) and v_types[i] in allowable_visits:
                flat_visits.append({'Patient': pid, 'VisitDate': v_dates[i], 'Freq': v_freqs[i] if i < len(v_freqs) else np.nan, 'HasSz': v_has_sz[i] if i < len(v_has_sz) else np.nan})

    vuniq = pd.DataFrame(flat_visits)
    if not vuniq.empty:
        vuniq['had_doc_freq'] = vuniq['Freq'].notna()
        vuniq = vuniq.groupby(['Patient', 'VisitDate']).agg(Freq_R1=('Freq', lambda x: x.mean(skipna=True)), Has_agg=('HasSz', lambda x: x.max(skipna=True)), had_doc_freq=('had_doc_freq', 'max')).reset_index()
        vuniq.loc[vuniq['Freq_R1'].isna() & (vuniq['Has_agg'] == 0.0), 'Freq_R1'] = 0.0
        patient_sz_freq = vuniq.groupby('Patient')['Freq_R1'].mean(skipna=True).reset_index().dropna(subset=['Freq_R1']).rename(columns={'Freq_R1': 'mean_sz_freq'})
        current_patients = set(patient_sz_freq['Patient'].unique())
    track_patients("Base Filter: Documented Seizure Frequency", before_pats, current_patients)

    # 4. SPIKE RATES
    patient_spikes = valid_sessions[valid_sessions['Patient'].isin(current_patients)].groupby('Patient').agg(total_spikes=("count_0_46", "sum"), total_duration=("Duration_sec", "sum"), mean_age_spike=("age", "mean"), median_age_sz=("age", "median")).reset_index()
    patient_spikes["spike_rate_per_hour"] = (patient_spikes["total_spikes"] / patient_spikes["total_duration"]) * 3600

    # 5. PROJECT FILTERS
    patient_df = pd.merge(patient_sz_freq, patient_spikes, on='Patient', how='inner').merge(pt_demo[['nlp_gender', 'epilepsy_type', 'epilepsy_specific']].reset_index(), on='Patient', how='inner')
    
    before_pats = current_patients.copy()
    patient_df = patient_df[patient_df['nlp_gender'].isin(['M', 'F'])]
    current_patients = set(patient_df['Patient'].unique())
    track_patients("Project Filter A: Valid Gender (M or F)", before_pats, current_patients)

    before_pats = current_patients.copy()
    patient_df['canonical_subtype'] = patient_df.apply(assign_canonical_subtype, axis=1)
    patient_df = patient_df[patient_df['canonical_subtype'].isin(['Focal', 'Generalized'])]
    current_patients = set(patient_df['Patient'].unique())
    track_patients("Project Filter B: Focal or Generalized Subtype", before_pats, current_patients)

    before_pats = current_patients.copy()
    patient_df = patient_df[~(patient_df['mean_age_spike'].isna() | patient_df['median_age_sz'].isna())]
    patient_df = patient_df[(patient_df['mean_age_spike'] >= 18.0) & (patient_df['median_age_sz'] >= 18.0)]
    current_patients = set(patient_df['Patient'].unique())
    track_patients("Project Filter C: Valid Adult Age (>= 18 years)", before_pats, current_patients)

    # AGE GROUPS & MEDIAN SPLITS
    bins, labels = [18, 40, 65, np.inf], ['18-39', '40-64', '65+']
    patient_df['age_group'] = pd.Categorical(pd.cut(patient_df['mean_age_spike'], bins=bins, labels=labels, right=False), categories=labels, ordered=True)
    
    med_spike = patient_df["spike_rate_per_hour"].median()
    med_sz = patient_df["mean_sz_freq"].median()
    patient_df["spike_rate_binary"] = (patient_df["spike_rate_per_hour"] >= med_spike).astype(int)
    patient_df["sz_freq_binary"] = (patient_df["mean_sz_freq"] >= med_sz).astype(int)

    # PRINT SUMMARY TABLES
    print_and_save_requested_stats(patient_df, valid_sessions, vuniq, save_dir)
    print_and_save_supplementary_table(patient_df, save_dir)

    # STANDARD LOGIT MODELS
    print_logit_results(patient_df, "spike_rate_binary", f"PRIMARY LOGIT: SPIKE RATE (≥{med_spike:.2f}/hr)")
    print_logit_results(patient_df, "sz_freq_binary", f"PRIMARY LOGIT: SEIZURE FREQUENCY (≥{med_sz:.2f}/mo)")
    
    # RUN FEMALE PRE- VS POST-MENOPAUSAL DETAILED COMPARISON WITH STATS
    compare_females_menopause(patient_df)

    # 6. RUN BOOTSTRAPS & FOREST PLOTS (PRIMARY)
    rows_spike = run_analysis(patient_df, "spike_rate_binary", f"SPIKE RATE (≥{med_spike:.2f}/hr)")
    rows_sz    = run_analysis(patient_df, "sz_freq_binary", f"SEIZURE FREQUENCY (≥{med_sz:.2f}/mo)")

    # 7. RUN BOOTSTRAPS & FOREST PLOTS (SUPPLEMENTARY MENOPAUSAL STRATIFICATION)
    rows_spike_meno = run_menopausal_stratified_analysis(patient_df, "spike_rate_binary", f"SPIKE RATE (≥{med_spike:.2f}/hr)")
    rows_sz_meno    = run_menopausal_stratified_analysis(patient_df, "sz_freq_binary", f"SEIZURE FREQUENCY (≥{med_sz:.2f}/mo)")

    forest_plot(
        rows_spike_meno,
        f"Spike Rate (≥{med_spike:.2f}/hr) Stratified by Menopausal Age Cutoff (51 Yrs)",
        (-1.0, 3.0),
        [-1.0, 0.0, 1.0, 2.0, 3.0],
        "",
        "← Fewer Spikes", 
        "More Spikes →",
        os.path.join(save_dir, "forest_spike_rate_boot_MENOPAUSE_STRATIFIED.png")
    )

    forest_plot(
        rows_sz_meno,
        f"Seizure Freq (≥{med_sz:.2f}/mo) Stratified by Menopausal Age Cutoff (51 Yrs)",
        (0.0, 2.5),
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        "",
        "← Fewer Seizures", 
        "More Seizures →",
        os.path.join(save_dir, "forest_seizure_freq_boot_MENOPAUSE_STRATIFIED.png")
    )
    
    print("\n===========================================================================")
    print("ALL PIPELINE TASKS & SUPPLEMENTARY MENOPAUSAL ANALYSES COMPLETE.")
    print("Check your save_dir for Table S1 and stratified forest plot PNGs!")
    print("===========================================================================\n")

if __name__ == "__main__":
    main()