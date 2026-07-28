# Bootstrapped binary forest plot of Logit models for spike rate and seizure frequency 
# Modeled by sex, epilepsy type, and age groups (18-39, 40-64, 65+) 
# with median split for binary spike rate and seizure frequency outcomes
# INCLUDES: Automated Spike Detector Validation & Men vs. Women Cohort Stratification

import os
import ast
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chi2_contingency


warnings.filterwarnings("ignore")
np.random.seed(42)

# ==========================================================
# 1. TRACKING, PARSING & SUBTYPE HELPERS
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
# 2. REPORTED SPIKE RESOLUTION & STATISTICAL HELPERS
# ==========================================================
def resolve_reported_spike_status(df):
    """
    Resolves clinical EEG report text into categorical spike status,
    matching the MATLAB hierarchy: main report column takes precedence,
    supplemented by lobe-specific ('jay_*') discharge columns.
    """
    df = df.copy()
    
    main_col = df['report_SPORADIC_EPILEPTIFORM_DISCHARGES'].fillna("").astype(str).str.lower().str.strip()
    focal = df['jay_focal_epi'].fillna("").astype(str).str.lower().str.strip()
    multi = df['jay_multifocal_epi'].fillna("").astype(str).str.lower().str.strip()
    gen = df['jay_gen_epi'].fillna("").astype(str).str.lower().str.strip()
    
    is_main_present = (main_col == "present")
    is_main_absent = (main_col == "absent")
    
    is_f_p, is_f_a = (focal == "present"), (focal == "absent")
    is_m_p, is_m_a = (multi == "present"), (multi == "absent")
    is_g_p, is_g_a = (gen == "present"), (gen == "absent")
    
    present_jay_any = is_f_p | is_m_p | is_g_p
    all_jay_absent = is_f_a & is_m_a & is_g_a
    blank_main = ~(is_main_present | is_main_absent)
    blank_jay_all = ~(is_f_p | is_f_a) & ~(is_m_p | is_m_a) & ~(is_g_p | is_g_a)
    
    status = pd.Series("unknown", index=df.index)
    status[is_main_present | present_jay_any] = "present"
    status[(all_jay_absent & blank_main) | (is_main_absent & blank_jay_all)] = "absent"
    
    df['ReportStatus'] = status
    return df

def cliffs_delta(x1, x2):
    """Computes Cliff's delta effect size for non-parametric distributions [-1, 1]."""
    x1 = np.asarray(x1)[np.isfinite(x1)]
    x2 = np.asarray(x2)[np.isfinite(x2)]
    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        return np.nan
    
    u_stat, _ = mannwhitneyu(x1, x2, alternative='two-sided')
    d = (2 * u_stat / (n1 * n2)) - 1
    return d

def boot_median_ci_val(series, n_boot=2000, alpha=0.05):
    """Returns (median, lo_ci, hi_ci) via bootstrapping."""
    s = np.asarray(series)[np.isfinite(series)]
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    med = np.median(s)
    boots = [np.median(np.random.choice(s, size=len(s), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return med, lo, hi

# ==========================================================
# 3. DETECTOR VALIDATION & SEX COMPARISON PIPELINE
# ==========================================================
def validate_detector_and_compare_sex(patient_df, valid_sessions, report_df, save_dir=None, n_boot=2000):
    print("\n" + "=" * 80)
    print("AUTOMATED SPIKE DETECTOR VALIDATION (VS. CLINICAL REPORTS)")
    print("=" * 80)
    
    # Merge valid sessions with clinical report columns
    session_report = pd.merge(
        valid_sessions, 
        report_df[['Patient', 'Session', 'report_SPORADIC_EPILEPTIFORM_DISCHARGES', 
                   'jay_focal_epi', 'jay_multifocal_epi', 'jay_gen_epi']],
        on=['Patient', 'Session'], 
        how='inner'
    )
    
    # Defensive check: ensure session-level rate is computed
    if 'spike_rate_per_hour' not in session_report.columns:
        session_report['spike_rate_per_hour'] = (session_report['count_0_46'] / session_report['Duration_sec']) * 3600
        
    session_report = resolve_reported_spike_status(session_report)
    
    cohort_pids = set(patient_df['Patient'].unique())
    eval_df = session_report[session_report['Patient'].isin(cohort_pids)].copy()
    
    rates_absent = eval_df[eval_df['ReportStatus'] == 'absent']['spike_rate_per_hour'].dropna()
    rates_present = eval_df[eval_df['ReportStatus'] == 'present']['spike_rate_per_hour'].dropna()
    
    u_stat, p_val = mannwhitneyu(rates_present, rates_absent, alternative='two-sided')
    delta = cliffs_delta(rates_present, rates_absent)
    
    med_abs, lo_abs, hi_abs = boot_median_ci_val(rates_absent, n_boot)
    med_pre, lo_pre, hi_pre = boot_median_ci_val(rates_present, n_boot)
    
    val_summary_data = [
        {"Report Status": "Clinically Absent", "N Sessions": len(rates_absent), "Median Spikes/Hr": med_abs, "95% CI Lower": lo_abs, "95% CI Upper": hi_abs},
        {"Report Status": "Clinically Present", "N Sessions": len(rates_present), "Median Spikes/Hr": med_pre, "95% CI Lower": lo_pre, "95% CI Upper": hi_pre}
    ]
    
    print(f"EEG Sessions evaluated: {len(eval_df):,}")
    print(f"  -> Clinically Absent (N={len(rates_absent):,}):  Median = {med_abs:.2f} [{lo_abs:.2f} - {hi_abs:.2f}] spikes/hr")
    print(f"  -> Clinically Present (N={len(rates_present):,}): Median = {med_pre:.2f} [{lo_pre:.2f} - {hi_pre:.2f}] spikes/hr")
    print(f"  -> Mann-Whitney U = {u_stat:,.1f}, p-value = {p_val:.3e}")
    print(f"  -> Cliff's Delta (d) = {delta:+.2f} (positive indicates automated rate is higher when reported present)\n")
    
    if save_dir:
        val_df = pd.DataFrame(val_summary_data)
        val_csv = os.path.join(save_dir, "detector_validation_summary.csv")
        val_df.to_csv(val_csv, index=False)
        print(f"-> Saved validation summary to: {val_csv}")

    # --- COHORT STRATIFICATION: MEN VS. WOMEN ---
    print("=" * 80)
    print("COHORT STRATIFICATION: MEN VS. WOMEN")
    print("=" * 80)
    
    males = patient_df[patient_df['nlp_gender'] == 'M']
    females = patient_df[patient_df['nlp_gender'] == 'F']
    
    m_spk, f_spk = males['spike_rate_per_hour'], females['spike_rate_per_hour']
    _, p_spk = mannwhitneyu(m_spk.dropna(), f_spk.dropna(), alternative='two-sided')
    d_spk = cliffs_delta(m_spk, f_spk)
    med_m_spk, lo_m_spk, hi_m_spk = boot_median_ci_val(m_spk, n_boot)
    med_f_spk, lo_f_spk, hi_f_spk = boot_median_ci_val(f_spk, n_boot)
    
    m_sz, f_sz = males['mean_sz_freq'], females['mean_sz_freq']
    _, p_sz = mannwhitneyu(m_sz.dropna(), f_sz.dropna(), alternative='two-sided')
    d_sz = cliffs_delta(m_sz, f_sz)
    med_m_sz, lo_m_sz, hi_m_sz = boot_median_ci_val(m_sz, n_boot)
    med_f_sz, lo_f_sz, hi_f_sz = boot_median_ci_val(f_sz, n_boot)
    
    _, p_age = mannwhitneyu(females['mean_age_spike'].dropna(), males['mean_age_spike'].dropna(), alternative='two-sided')
    d_age = cliffs_delta(males['mean_age_spike'], females['mean_age_spike'])
    
    sex_comp_data = [
        {"Metric": "Patient Count (N)", "Women (F)": f"{len(females):,}", "Men (M)": f"{len(males):,}", "p-value": "-", "Cliff's d": "-"},
        {"Metric": "Age at Spike Evaluation Median (IQR)", 
         "Women (F)": f"{females['mean_age_spike'].median():.1f} ({females['mean_age_spike'].quantile(0.25):.1f}-{females['mean_age_spike'].quantile(0.75):.1f})",
         "Men (M)": f"{males['mean_age_spike'].median():.1f} ({males['mean_age_spike'].quantile(0.25):.1f}-{males['mean_age_spike'].quantile(0.75):.1f})",
         "p-value": f"{p_age:.3f}", "Cliff's d": f"{d_age:+.2f}"},
        {"Metric": "Spike Rate (spikes/hr) Median [95% CI]", 
         "Women (F)": f"{med_f_spk:.2f} [{lo_f_spk:.2f}-{hi_f_spk:.2f}]", 
         "Men (M)": f"{med_m_spk:.2f} [{lo_m_spk:.2f}-{hi_m_spk:.2f}]", 
         "p-value": f"{p_spk:.3f}", "Cliff's d": f"{d_spk:+.2f}"},
        {"Metric": "Seizure Freq (sz/month) Median [95% CI]", 
         "Women (F)": f"{med_f_sz:.2f} [{lo_f_sz:.2f}-{hi_f_sz:.2f}]", 
         "Men (M)": f"{med_m_sz:.2f} [{lo_m_sz:.2f}-{hi_m_sz:.2f}]", 
         "p-value": f"{p_sz:.3f}", "Cliff's d": f"{d_sz:+.2f}"},
        {"Metric": "% High Spike Rate (≥ Cohort Median)", 
         "Women (F)": f"{(females['spike_rate_binary'].mean()*100):.1f}%", 
         "Men (M)": f"{(males['spike_rate_binary'].mean()*100):.1f}%", 
         "p-value": "-", "Cliff's d": "-"},
        {"Metric": "% High Seizure Freq (≥ Cohort Median)", 
         "Women (F)": f"{(females['sz_freq_binary'].mean()*100):.1f}%", 
         "Men (M)": f"{(males['sz_freq_binary'].mean()*100):.1f}%", 
         "p-value": "-", "Cliff's d": "-"}
    ]
    
    comp_df = pd.DataFrame(sex_comp_data)
    print(comp_df.to_string(index=False))
    print("=" * 80 + "\n")
    
    if save_dir:
        comp_csv = os.path.join(save_dir, "sex_stratification_comparison.csv")
        comp_df.to_csv(comp_csv, index=False)
        print(f"-> Saved sex stratification table to: {comp_csv}\n")
        
    return eval_df, comp_df

# ==========================================================
# 4. COHORT STATS GENERATOR & PRETTY IMAGE/TABLE SAVER
# ==========================================================
def print_and_save_requested_stats(patient_df, valid_sessions, vuniq, save_dir, n_boot=2000):
    print("\n" + "=" * 75)
    print("FINAL COHORT SUMMARY STATISTICS")
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
    n_general = len(patient_df[patient_df['canonical_subtype'] == 'General'])
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
        {"Category": "Epilepsy Subtype", "Metric": "Generalized N (%)", "Value": f"{n_general:,} ({(n_general/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
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
    html_style = """
    <style>
        table { border-collapse: collapse; width: 80%; font-family: Arial, sans-serif; margin: 20px 0; }
        th, td { border: 1px solid #dddddd; text-align: left; padding: 10px; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
    </style>
    """
    with open(html_path, "w") as f:
        f.write(f"<html><head><title>Table 1 Summary</title>{html_style}</head><body>")
        f.write("<h2>Final Cohort Summary Statistics (Table 1)</h2>")
        f.write(summary_df.to_html(index=False))
        f.write("</body></html>")

    png_path = os.path.join(save_dir, "final_cohort_table1_summary.png")
    fig, ax = plt.subplots(figsize=(15, len(summary_df) * 0.40 + 1.2))
    ax.axis("off")
    
    col_labels = summary_df.columns.tolist()
    cell_text = summary_df.values.tolist()
    custom_widths = [0.18, 0.46, 0.18, 0.18]
    
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=custom_widths,
        loc="upper center",
        cellLoc="left"
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row_idx == 0:
            cell.set_facecolor("#1A365D")
            cell.set_text_props(color="white", fontweight="bold")
            if col_idx in [2, 3]:
                cell.set_text_props(ha="right")
        else:
            if row_idx % 2 == 0:
                cell.set_facecolor("#F8FAFC")
            else:
                cell.set_facecolor("white")
            if col_idx in [2, 3]:
                cell.set_text_props(ha="right")
                
    ax.set_title("Final Cohort Baseline Characteristics (Table 1)", 
                 fontweight="bold", fontsize=16, y=0.98, pad=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"-> Saved clean summary outputs to:\n   1. {png_path}\n   2. {csv_path}\n   3. {html_path}\n")

# ==========================================================
# 5. LOGISTIC BOOTSTRAP HELPERS
# ==========================================================
def bootstrap_regression_coeffs(df, outcome_col, formula_vars, n_boot=5000):
    df = df.copy()
    print(f"\nChecking {outcome_col}")
    print(df[outcome_col].value_counts())

    df = df[np.isfinite(df[outcome_col])]
    cols_to_dummy = [outcome_col, "nlp_gender", "canonical_subtype", "age_group"]
    reg_df = pd.get_dummies(df[cols_to_dummy], drop_first=True).astype(float)

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
                model = sm.Logit(y, X).fit(disp=0)
                beta = model.params[var]
                if np.isfinite(beta):
                    coeffs.append(beta)
            except Exception:
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
    
    formula_vars = ['nlp_gender_M', 'canonical_subtype_General', 'age_group_40-64', 'age_group_65+']
    coeffs = bootstrap_regression_coeffs(patient_df, outcome_col, formula_vars)
    
    rows = []
    
    # Section 1: Sex
    m_res = coeffs['nlp_gender_M']
    female_n = len(patient_df[patient_df['nlp_gender']=='F'])
    rows.append(dict(section="Sex", label="Female", n=female_n, is_reference=True, diff=0, lo=0, hi=0, p=None))
    rows.append(dict(section="Sex", label="Male", n=len(patient_df[patient_df['nlp_gender']=='M']), is_reference=False, diff=m_res[0], lo=m_res[1], hi=m_res[2], p=m_res[3]))

    # Section 2: Epilepsy Type
    g_res = coeffs['canonical_subtype_General']
    focal_n = len(patient_df[patient_df['canonical_subtype']=='Focal'])
    rows.append(dict(section="Epilepsy Type", label="Focal", n=focal_n, is_reference=True, diff=0, lo=0, hi=0, p=None))
    rows.append(dict(section="Epilepsy Type", label="General", n=len(patient_df[patient_df['canonical_subtype']=='General']), is_reference=False, diff=g_res[0], lo=g_res[1], hi=g_res[2], p=g_res[3]))

    # Section 3: Age Group
    age1_n = len(patient_df[patient_df['age_group']=='18-39'])
    rows.append(dict(section="Age Group", label="18-39 years", n=age1_n, is_reference=True, diff=0, lo=0, hi=0, p=None))
    
    a2_res = coeffs['age_group_40-64']
    rows.append(dict(section="Age Group", label="40-64 years", n=len(patient_df[patient_df['age_group']=='40-64']), is_reference=False, diff=a2_res[0], lo=a2_res[1], hi=a2_res[2], p=a2_res[3]))
    
    a3_res = coeffs['age_group_65+']
    rows.append(dict(section="Age Group", label="65+ years", n=len(patient_df[patient_df['age_group']=='65+']), is_reference=False, diff=a3_res[0], lo=a3_res[1], hi=a3_res[2], p=a3_res[3]))

    return rows

# ==========================================================
# 6. PLOTTING HELPERS
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
    ax.text(COL_EST, top, "Coefficient (95% CI)", fontweight="bold", va="center")
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
            ax.scatter(map_x(0), y, s=50, marker="s", color="black")
            ax.text(COL_P, y, "Ref.", va="center")
        else:
            d = row["diff"]
            lo = row["lo"]
            hi = row["hi"]

            ax.text(COL_EST, y, f"{d:+.2f} ({lo:+.2f}, {hi:+.2f})", va="center")
            ax.plot([map_x(lo), map_x(hi)], [y, y], lw=2, color="black")
            ax.scatter(map_x(d), y, s=50, marker="s", color="black")

            ptxt = "<0.001" if row["p"] < 0.001 else f"{row['p']:.3f}"
            ax.text(COL_P, y, ptxt, va="center")

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
# 7. STANDARD LOGISTIC REGRESSION RESULTS
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

    model = sm.Logit(y, X).fit()

    print(model.summary())
    print("\n95% CI (Log-Odds)")
    print(model.conf_int())
    
    print("\nOdds Ratios (Exponentiated Coefficients)")
    print(np.exp(model.params))

    return model

# =========================================================================
# AUTOMATED DETECTOR PERFORMANCE & SEX FALSE POSITIVE RATE (FPR) PIPELINE
# =========================================================================
def evaluate_detector_performance_and_sex_fpr(patient_df, valid_sessions, report_df, save_dir=None, n_boot=2000):
    print("\n" + "=" * 85)
    print("AUTOMATED DETECTOR PERFORMANCE & SEX FALSE POSITIVE RATE (FPR) ANALYSIS")
    print("=" * 85)
    
    # 1. Merge sessions with report text and demographic gender
    session_report = pd.merge(
        valid_sessions, 
        report_df[['Patient', 'Session', 'report_SPORADIC_EPILEPTIFORM_DISCHARGES', 
                   'jay_focal_epi', 'jay_multifocal_epi', 'jay_gen_epi']],
        on=['Patient', 'Session'], 
        how='inner'
    )
    
    # Ensure continuous rate exists
    if 'spike_rate_per_hour' not in session_report.columns:
        session_report['spike_rate_per_hour'] = (session_report['count_0_46'] / session_report['Duration_sec']) * 3600
        
    session_report = resolve_reported_spike_status(session_report)
    
    # Restrict to final cleaned patient cohort and valid M/F gender
    cohort_pids = set(patient_df['Patient'].unique())
    eval_df = session_report[session_report['Patient'].isin(cohort_pids)].copy()
    eval_df = pd.merge(eval_df, patient_df[['Patient', 'nlp_gender']], on='Patient', how='inner')
    eval_df = eval_df[eval_df['nlp_gender'].isin(['M', 'F'])].copy()
    
    # Separate into clinical ground truth groups
    absent_df = eval_df[eval_df['ReportStatus'] == 'absent'].copy()
    present_df = eval_df[eval_df['ReportStatus'] == 'present'].copy()
    
    # Dynamic background noise threshold (median rate of clean EEGs)
    clean_median_thresh = absent_df['spike_rate_per_hour'].median()
    
    # =========================================================================
    # PART A: OVERALL DIAGNOSTIC PERFORMANCE (SENSITIVITY & SPECIFICITY)
    # =========================================================================
    print("--- PART A: OVERALL DETECTOR DIAGNOSTIC ACCURACY ---")
    print(f"Total Evaluated EEGs: {len(eval_df):,} (Clean Absent: {len(absent_df):,} | Active Present: {len(present_df):,})")
    
    # Threshold 1: Strict Detection (> 0.0 spikes/hr)
    tp_strict = (present_df['spike_rate_per_hour'] > 0).sum()
    fn_strict = len(present_df) - tp_strict
    fp_strict = (absent_df['spike_rate_per_hour'] > 0).sum()
    tn_strict = len(absent_df) - fp_strict
    
    sens_strict = (tp_strict / len(present_df)) * 100
    spec_strict = (tn_strict / len(absent_df)) * 100
    fpr_strict  = (fp_strict / len(absent_df)) * 100
    
    # Threshold 2: Background Noise Adjusted (>= Clean EEG Median)
    tp_adj = (present_df['spike_rate_per_hour'] >= clean_median_thresh).sum()
    fp_adj = (absent_df['spike_rate_per_hour'] >= clean_median_thresh).sum()
    
    sens_adj = (tp_adj / len(present_df)) * 100
    fpr_adj  = (fp_adj / len(absent_df)) * 100
    spec_adj = 100 - fpr_adj
    
    print(f"  [Threshold > 0.00/hr]: Sensitivity = {sens_strict:.1f}% | Specificity = {spec_strict:.1f}% | FPR = {fpr_strict:.1f}%")
    print(f"  [Threshold ≥ {clean_median_thresh:.2f}/hr (Clean Median)]: Sensitivity = {sens_adj:.1f}% | Specificity = {spec_adj:.1f}% | FPR = {fpr_adj:.1f}%\n")
    
    # =========================================================================
    # PART B: FALSE POSITIVE COMPARISON — MEN VS. WOMEN
    # =========================================================================
    print("--- PART B: FALSE POSITIVE DISPARITIES (MEN VS. WOMEN ON CLEAN EEGs) ---")
    
    m_abs = absent_df[absent_df['nlp_gender'] == 'M']
    f_abs = absent_df[absent_df['nlp_gender'] == 'F']
    
    # 1. Continuous False Positive Burden (Noise Rate in Spikes/Hr)
    m_fp_rates = m_abs['spike_rate_per_hour'].dropna()
    f_fp_rates = f_abs['spike_rate_per_hour'].dropna()
    
    u_fp, p_fp_cont = mannwhitneyu(m_fp_rates, f_fp_rates, alternative='two-sided')
    d_fp = cliffs_delta(m_fp_rates, f_fp_rates)
    
    med_m_fp, lo_m_fp, hi_m_fp = boot_median_ci_val(m_fp_rates, n_boot)
    med_f_fp, lo_f_fp, hi_f_fp = boot_median_ci_val(f_fp_rates, n_boot)
    
    # 2. Binary False Positive Rate (FPR at Strict > 0 Threshold)
    m_fp_strict_cnt = (m_abs['spike_rate_per_hour'] > 0).sum()
    f_fp_strict_cnt = (f_abs['spike_rate_per_hour'] > 0).sum()
    
    m_fpr_strict = (m_fp_strict_cnt / len(m_abs)) * 100
    f_fpr_strict = (f_fp_strict_cnt / len(f_abs)) * 100
    
    # Chi-Square Test for Strict FPR
    contingency_strict = [
        [m_fp_strict_cnt, len(m_abs) - m_fp_strict_cnt],
        [f_fp_strict_cnt, len(f_abs) - f_fp_strict_cnt]
    ]
    _, p_fp_bin_strict, _, _ = chi2_contingency(contingency_strict)
    
    # 3. Binary False Positive Rate (FPR at Clean Median Threshold)
    m_fp_adj_cnt = (m_abs['spike_rate_per_hour'] >= clean_median_thresh).sum()
    f_fp_adj_cnt = (f_abs['spike_rate_per_hour'] >= clean_median_thresh).sum()
    
    m_fpr_adj = (m_fp_adj_cnt / len(m_abs)) * 100
    f_fpr_adj = (f_fp_adj_cnt / len(f_abs)) * 100
    
    # Chi-Square Test for Adjusted FPR
    contingency_adj = [
        [m_fp_adj_cnt, len(m_abs) - m_fp_adj_cnt],
        [f_fp_adj_cnt, len(f_abs) - f_fp_adj_cnt]
    ]
    _, p_fp_bin_adj, _, _ = chi2_contingency(contingency_adj)
    
    # Build Structured Output Table
    fpr_table_data = [
        {"Metric": "Clean EEG Sessions Evaluated (N)", "Women (F)": f"{len(f_abs):,}", "Men (M)": f"{len(m_abs):,}", "p-value": "-", "Effect Size": "-"},
        {"Metric": "Continuous FP Rate (spikes/hr) Median [95% CI]", 
         "Women (F)": f"{med_f_fp:.2f} [{lo_f_fp:.2f}-{hi_f_fp:.2f}]", 
         "Men (M)": f"{med_m_fp:.2f} [{lo_m_fp:.2f}-{hi_m_fp:.2f}]", 
         "p-value": f"{p_fp_cont:.3f}", "Effect Size": f"d = {d_fp:+.2f}"},
        {"Metric": "Binary FPR (Strict > 0.00 spikes/hr)", 
         "Women (F)": f"{f_fpr_strict:.1f}% ({f_fp_strict_cnt:,})", 
         "Men (M)": f"{m_fpr_strict:.1f}% ({m_fp_strict_cnt:,})", 
         "p-value": f"{p_fp_bin_strict:.3f}", "Effect Size": f"Δ = {(m_fpr_strict - f_fpr_strict):+.1f}%"},
        {"Metric": f"Binary FPR (Adjusted ≥ {clean_median_thresh:.2f}/hr)", 
         "Women (F)": f"{f_fpr_adj:.1f}% ({f_fp_adj_cnt:,})", 
         "Men (M)": f"{m_fpr_adj:.1f}% ({m_fp_adj_cnt:,})", 
         "p-value": f"{p_fp_bin_adj:.3f}", "Effect Size": f"Δ = {(m_fpr_adj - f_fpr_adj):+.1f}%"},
        {"Metric": "True Positive Rate / Sensitivity (> 0.00/hr)", 
         "Women (F)": f"{((present_df[present_df['nlp_gender']=='F']['spike_rate_per_hour']>0).mean()*100):.1f}%", 
         "Men (M)": f"{((present_df[present_df['nlp_gender']=='M']['spike_rate_per_hour']>0).mean()*100):.1f}%", 
         "p-value": "-", "Effect Size": "-"}
    ]
    
    fpr_df = pd.DataFrame(fpr_table_data)
    print(fpr_df.to_string(index=False))
    print("=" * 85 + "\n")
    
    if save_dir:
        csv_out = os.path.join(save_dir, "detector_sex_fpr_comparison.csv")
        fpr_df.to_csv(csv_out, index=False)
        print(f"-> Saved sex false-positive analysis to: {csv_out}\n")
        
    return fpr_df

# ==========================================================
# 8. MAIN PIPELINE
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
    
    # Calculate session-level spike rate immediately (matches MATLAB SpikeRate_perHour)
    spike_df['spike_rate_per_hour'] = (spike_df['count_0_46'] / spike_df['Duration_sec']) * 3600

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

    before_pats = current_patients.copy()
    missing_age_mask = patient_df['mean_age_spike'].isna() | patient_df['median_age_sz'].isna()
    patient_df = patient_df[~missing_age_mask]
    current_patients = set(patient_df['Patient'].unique())
    track_patients("Project Filter C1: Remove Missing Age (NaN)", before_pats, current_patients)

    before_pats = current_patients.copy()
    invalid_age_mask = (patient_df['mean_age_spike'] < 18.0) | (patient_df['median_age_sz'] < 18.0)
    patient_df = patient_df[~invalid_age_mask]
    current_patients = set(patient_df['Patient'].unique())
    track_patients("Project Filter C2: Keep Valid Adult Age (>= 18 years)", before_pats, current_patients)

    # CREATE CATEGORICAL AGE GROUP (18-39 as reference)
    bins = [18, 40, 65, np.inf] # [18, 40) = 18-39.99
    labels = ['18-39', '40-64', '65+']
    patient_df['age_group'] = pd.cut(patient_df['mean_age_spike'], bins=bins, labels=labels, right=False)
    patient_df['age_group'] = pd.Categorical(patient_df['age_group'], categories=labels, ordered=True)

    # BINARY CONVERSION BASED ON DYNAMIC MEDIANS
    med_spike = patient_df["spike_rate_per_hour"].median()
    med_sz = patient_df["mean_sz_freq"].median()
    
    patient_df["spike_rate_binary"] = (patient_df["spike_rate_per_hour"] >= med_spike).astype(int)
    patient_df["sz_freq_binary"] = (patient_df["mean_sz_freq"] >= med_sz).astype(int)
    
    # 6. EXECUTE VALIDATION & SEX COMPARISON STEPS
    eval_df, sex_comp_df = validate_detector_and_compare_sex(patient_df, valid_sessions, clinical_df, save_dir=save_dir)
    fpr_comp_df = evaluate_detector_performance_and_sex_fpr(patient_df, valid_sessions, clinical_df, save_dir=save_dir)

    # =========================================================================
    # 6. EXECUTE VALIDATION & SEX COMPARISON STEP
    # =========================================================================
    eval_df, sex_comp_df = validate_detector_and_compare_sex(patient_df, valid_sessions, clinical_df, save_dir=save_dir)

    # # =========================================================================
    # # 7. PRINT PRETTY TABLE TO TERMINAL AND SAVE AS PNG/CSV/HTML
    # # =========================================================================
    # print_and_save_requested_stats(patient_df, valid_sessions, vuniq, save_dir)

    # print("=" * 75)
    # print(f"FINAL PATIENT COHORT SIZE: {len(current_patients):,} patients")
    # print(f"  -> Median Spike Rate Split:     {med_spike:.2f} / hr")
    # print(f"  -> Median Seizure Freq Split:   {med_sz:.2f} / mo")
    # print("=" * 75)

    # print("\n======================================================")
    # print("               AGE GROUP DISTRIBUTION                 ")
    # print("======================================================")
    # print(patient_df['age_group'].value_counts().sort_index())
    # print("======================================================\n")

    # # 8. STANDARD LOGISTIC REGRESSION RESULTS
    # print_logit_results(patient_df, "spike_rate_binary", f"LOGISTIC REGRESSION: SPIKE RATE (≥{med_spike:.2f}/hr)")
    # print_logit_results(patient_df, "sz_freq_binary", f"LOGISTIC REGRESSION: SEIZURE FREQUENCY (≥{med_sz:.2f}/mo)")

    # # 9. RUN BOOTSTRAP ANALYSIS
    # rows_spike = run_analysis(patient_df, "spike_rate_binary", f"SPIKE RATE (≥{med_spike:.2f}/hr)")
    # rows_sz    = run_analysis(patient_df, "sz_freq_binary", f"SEIZURE FREQUENCY (≥{med_sz:.2f}/mo)")

    # # 10. GENERATE FOREST PLOTS
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