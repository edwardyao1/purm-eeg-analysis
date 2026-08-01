# Bootstrapped binary forest plot of Logit models for spike rate and seizure frequency 
# Modeled by sex, epilepsy type, and age groups (18-39, 40-64, 65+) 
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

    # Calculate exact counts and percentages
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

    # Build structured list for DataFrame
    table_data = [
        {"Category": "Total Patients", "Metric": "N", "Value": f"{n_patients:,}", "Bootstrapped 95% CI": ""},
        {"Category": "Sex", "Metric": "Men N (%)", "Value": f"{n_males:,} ({(n_males/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "Sex", "Metric": "Women N (%)", "Value": f"{n_females:,} ({(n_females/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "Epilepsy Subtype", "Metric": "Focal Lobe N (%)", "Value": f"{n_focal:,} ({(n_focal/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "Epilepsy Subtype", "Metric": "Generalized N (%)", "Value": f"{n_generalized:,} ({(n_generalized/n_patients)*100:.1f}%)", "Bootstrapped 95% CI": ""},
        {"Category": "Age", "Metric": "Age at first clinic visit Median (IQR)", "Value": median_iqr(patient_df['age_at_first_visit'], 1), "Bootstrapped 95% CI": ""},
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

    # 1. Print Text Table to Terminal
    print(summary_df.to_string(index=False))
    print("=" * 75 + "\n")

    # 2. Save as CSV
    csv_path = os.path.join(save_dir, "final_cohort_table1_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    
    # # 3. Save as Styled HTML
    # html_path = os.path.join(save_dir, "final_cohort_table1_summary.html")
    # html_style = """
    # <style>
    #     table { border-collapse: collapse; width: 80%; font-family: Arial, sans-serif; margin: 20px 0; }
    #     th, td { border: 1px solid #dddddd; text-align: left; padding: 10px; }
    #     th { background-color: #f2f2f2; font-weight: bold; }
    #     tr:nth-child(even) { background-color: #f9f9f9; }
    # </style>
    # """
    # with open(html_path, "w") as f:
    #     f.write(f"<html><head><title>Table 1 Summary</title>{html_style}</head><body>")
    #     f.write("<h2>Final Cohort Summary Statistics (Table 1)</h2>")
    #     f.write(summary_df.to_html(index=False))
    #     f.write("</body></html>")

    # 4. Save as PNG image
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
            cell.set_facecolor("#000000")
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

def print_sex_stratified_descriptives(patient_df):
    """Prints the median and IQR of outcomes stratified by sex."""
    print("\n" + "=" * 75)
    print("OUTCOMES STRATIFIED BY SEX: MEDIAN (IQR)")
    print("=" * 75)
    
    for gender, label in [('M', 'Men'), ('F', 'Women')]:
        sub_df = patient_df[patient_df['nlp_gender'] == gender]
        spk = sub_df['spike_rate_per_hour'].dropna()
        sz = sub_df['mean_sz_freq'].dropna()
        
        if len(spk) > 0:
            spk_med, spk_q25, spk_q75 = spk.median(), spk.quantile(0.25), spk.quantile(0.75)
            spk_str = f"{spk_med:.2f} ({spk_q25:.2f} - {spk_q75:.2f})"
        else:
            spk_str = "N/A"
            
        if len(sz) > 0:
            sz_med, sz_q25, sz_q75 = sz.median(), sz.quantile(0.25), sz.quantile(0.75)
            sz_str = f"{sz_med:.2f} ({sz_q25:.2f} - {sz_q75:.2f})"
        else:
            sz_str = "N/A"
            
        print(f"  {label} (N = {len(sub_df)}):")
        print(f"    -> Spike Rate (spikes/hr): {spk_str}")
        print(f"    -> Seizure Freq (sz/mo):   {sz_str}\n")

# ==========================================================
# LOGISTIC BOOTSTRAP HELPERS
# ==========================================================
def bootstrap_regression_coeffs(df, outcome_col, formula_vars, n_boot=5000):
    df = df.copy()

    np.random.seed(42)

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
    print(f"BOOTSTRAP LOGIT REGRESSION (ODDS RATIOS) — {outcome_label}")
    print(f"===========================================================================")
    
    formula_vars = ['nlp_gender_M', 'canonical_subtype_Generalized', 'age_group_40-64', 'age_group_65+']
    coeffs = bootstrap_regression_coeffs(patient_df, outcome_col, formula_vars)
    
    rows = []
    
    # Exponentiate log-odds to get Odds Ratios (OR = e^beta) while keeping p-value exact
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
    age1_n = len(patient_df[patient_df['age_group']=='18-39'])
    rows.append(dict(section="Age Group", label="18-39 years", n=age1_n, is_reference=True, diff=1.0, lo=1.0, hi=1.0, p=None))
    
    a2_res = to_or(coeffs['age_group_40-64'])
    rows.append(dict(section="Age Group", label="40-64 years", n=len(patient_df[patient_df['age_group']=='40-64']), is_reference=False, diff=a2_res[0], lo=a2_res[1], hi=a2_res[2], p=a2_res[3]))
    
    a3_res = to_or(coeffs['age_group_65+'])
    rows.append(dict(section="Age Group", label="65+ years", n=len(patient_df[patient_df['age_group']=='65+']), is_reference=False, diff=a3_res[0], lo=a3_res[1], hi=a3_res[2], p=a3_res[3]))

    return rows

# ==========================================================
# PLOTTING HELPERS (CUSTOM FOREST PLOT WITH LARGE FONTS)
# ==========================================================
def forest_plot(rows, title, x_lim, x_ticks, x_label, left_dir_label, right_dir_label, out_path):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    sections = []
    for r in rows:
        if r["section"] not in sections:
            sections.append(r["section"])

    # Bounds for the plot and p-value
    COL_SUB = 0.02
    COL_N = 0.25   
    COL_EST = 0.41 
    PLOT_L = 0.59  
    PLOT_R = 0.86  
    COL_P = 0.91  

    def map_x(v):
        frac = (v - x_lim[0]) / (x_lim[1] - x_lim[0])
        return PLOT_L + frac * (PLOT_R - PLOT_L)

    # Spacing metrics
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

    # Top row headers
    ax.text(COL_SUB, top, "Subgroup / Variable", fontweight="bold", va="center", fontsize=14)
    ax.text(COL_N, top, "Patients (N)", fontweight="bold", va="center", fontsize=14)
    ax.text(COL_EST, top, "Odds Ratio (95% CI)", fontweight="bold", va="center", fontsize=14) 
    ax.text(COL_P, top, "P-Value", fontweight="bold", va="center", fontsize=14)
    ax.text(map_x(1.0), top, "Odds Ratio (95% CI)", fontweight="bold", ha="center", va="center", fontsize=14)

    # Top black dividing line
    ax.plot([0, 1], [top - 0.3, top - 0.3], color="black")

    for sec in sections:
        # Category headers 
        ax.text(COL_SUB, ypos[(sec, "__header__")], sec, fontsize=15, fontweight="bold", va="center")

    for row in rows:
        y_loc = ypos[(row["section"], row["label"])]
        
        # Dynamically append (Ref.) to reference categories
        label_text = f'{row["label"]} (Ref.)' if row["is_reference"] else row["label"]
        
        # Row Labels and N Counts 
        ax.text(COL_SUB + 0.04, y_loc, label_text, va="center", fontsize=14)
        ax.text(COL_N, y_loc, str(row["n"]), va="center", fontsize=14)

        if row["is_reference"]:
            # Reference Text 
            ax.text(COL_EST, y_loc, "Ref.", va="center", fontsize=14)
            ax.scatter(map_x(1.0), y_loc, s=50, marker="s", color="black")
            ax.text(COL_P, y_loc, "Ref.", va="center", fontsize=14)
        else:
            d = row["diff"]
            lo = row["lo"]
            hi = row["hi"]

            # Output Values 
            ax.text(COL_EST, y_loc, f"{d:.2f} ({lo:.2f}-{hi:.2f})", va="center", fontsize=14)
            ax.plot([map_x(lo), map_x(hi)], [y_loc, y_loc], lw=2, color="black")
            ax.scatter(map_x(d), y_loc, s=50, marker="s", color="black")

            ptxt = "<0.001" if row["p"] < 0.001 else f"{row['p']:.3f}"
            ax.text(COL_P, y_loc, ptxt, va="center", fontsize=14)

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
        ax.text(x_loc, axis_y - 0.30, str(tick), ha="center", va="top", fontsize=12) # Tick numbers increased

    # =========================================================================
    # DOUBLE-SIDED DIRECTIONAL ARROW (<--->)
    # =========================================================================
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

    # Overall plot Title 
    plt.title(title, fontweight="bold", fontsize=17, pad=25)
    
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
    reg_df["canonical_subtype_Generalized"] = (reg_df["canonical_subtype"] == "Generalized").astype(float)
    reg_df["age_group_40-64"] = (reg_df["age_group"] == "40-64").astype(float)
    reg_df["age_group_65+"] = (reg_df["age_group"] == "65+").astype(float)

    X = reg_df[["nlp_gender_M", "canonical_subtype_Generalized", "age_group_40-64", "age_group_65+"]].astype(float)
    X = sm.add_constant(X)
    
    y = reg_df[outcome_col].astype(float)

    # Fit the model
    model = sm.Logit(y, X).fit(disp=0) # disp=0 hides the iterative optimization text

    # Extract Log-Odds and exponentiate to get Odds Ratios
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues

    or_df = pd.DataFrame({
        'Odds Ratio': np.exp(params),
        'CI 2.5%': np.exp(conf[0]),
        'CI 97.5%': np.exp(conf[1]),
        'P-Value': pvals
    })

    # Format the numbers for a clean printout
    or_df['Odds Ratio'] = or_df['Odds Ratio'].apply(lambda x: f"{x:.3f}")
    or_df['CI 2.5%'] = or_df['CI 2.5%'].apply(lambda x: f"{x:.3f}")
    or_df['CI 97.5%'] = or_df['CI 97.5%'].apply(lambda x: f"{x:.3f}")
    or_df['P-Value'] = or_df['P-Value'].apply(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")

    print("\n--- Standard Logistic Regression: ODDS RATIOS ---")
    print(or_df.to_string())
    print("-" * 60)
    
    # Optional: You can uncomment the line below if you still want the massive statsmodels dump
    print("\n--- Full Model Summary ---")
    print(model.summary())

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
        # Convert VisitDate to datetime early so we can do accurate time deltas
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

    track_patients("Base Filter: Documented Seizure Frequency (Primary Cohort)", before_pats, current_patients)

    # 4. CALCULATE PATIENT SPIKE RATES & AGES
    patient_spikes = valid_sessions[valid_sessions['Patient'].isin(current_patients)].groupby('Patient').agg(
        total_spikes=("count_0_46", "sum"),
        total_duration=("Duration_sec", "sum"),
        mean_age_spike=("age", "mean"),
        median_age_sz=("age", "median")
    ).reset_index()
    patient_spikes["spike_rate_per_hour"] = (patient_spikes["total_spikes"] / patient_spikes["total_duration"]) * 3600

    # 4.5. CALCULATE AGE AT FIRST VISIT
    bday_df = clinical_df[['Patient', 'deid_birth_date']].drop_duplicates('Patient')
    first_v_df = vuniq[vuniq['Patient'].isin(current_patients)].groupby('Patient')['VisitDate'].min().reset_index()
    first_v_df = first_v_df.merge(bday_df, on='Patient', how='inner')
    first_v_df['deid_birth_date'] = pd.to_datetime(first_v_df['deid_birth_date'], errors='coerce')
    first_v_df['age_at_first_visit'] = (first_v_df['VisitDate'] - first_v_df['deid_birth_date']).dt.days / 365.25

    # 5. PROJECT SPECIFIC FILTERS & AGE DIAGNOSTICS
    print("======================================================")
    print("               PROJECT SPECIFIC FILTERS               ")
    print("======================================================\n")
    
    patient_df = pd.merge(patient_sz_freq, patient_spikes, on='Patient', how='inner')
    patient_df = patient_df.merge(pt_demo[['nlp_gender', 'epilepsy_type', 'epilepsy_specific']].reset_index(), on='Patient', how='inner')
    patient_df = patient_df.merge(first_v_df[['Patient', 'age_at_first_visit']], on='Patient', how='left')
    
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
    missing_age_mask = patient_df['mean_age_spike'].isna() | patient_df['median_age_sz'].isna()
    patient_df = patient_df[~missing_age_mask]
    current_patients = set(patient_df['Patient'].unique())
    track_patients("Project Filter C1: Remove Missing Age (NaN)", before_pats, current_patients)

    before_pats = current_patients.copy()
    invalid_age_mask = (patient_df['mean_age_spike'] < 18.0) | (patient_df['median_age_sz'] < 18.0)
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

    # Print table and save as PNG/CSV/HTML
    print_and_save_requested_stats(patient_df, valid_sessions, vuniq, save_dir)
    
    # Print medians and IQRs for males and females
    print_sex_stratified_descriptives(patient_df)

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

    # 6. STANDARD LOGISTIC REGRESSION RESULTS
    print_logit_results(patient_df, "spike_rate_binary", f"LOGISTIC REGRESSION: SPIKE RATE (≥{med_spike:.2f}/hr)")
    print_logit_results(patient_df, "sz_freq_binary", f"LOGISTIC REGRESSION: SEIZURE FREQUENCY (≥{med_sz:.2f}/mo)")

    # 7. RUN BOOTSTRAP ANALYSIS (NOW EXPONENTIATED TO ORs)
    rows_spike = run_analysis(patient_df, "spike_rate_binary", f"SPIKE RATE (≥{med_spike:.2f}/hr)")
    rows_sz    = run_analysis(patient_df, "sz_freq_binary", f"SEIZURE FREQUENCY (≥{med_sz:.2f}/mo)")

    # 8. GENERATE FOREST PLOTS (WITH DIRECTIONAL ARROWS & OR AXES)
    print("\n--- Generating Forest Plots ---")
    
    forest_plot(
        rows_spike,
        f"Bootstrapped Odds Ratios: Spike Rate (≥{med_spike:.2f}/hr) ~ Sex + Epilepsy Type + Age Group",
        (0.0, 2.0),
        [0.0, 0.5, 1.0, 1.5, 2.0],
        "", # Bottom axis label is empty
        "← Fewer Spikes",
        "More Spikes →",
        os.path.join(save_dir, "forest_spike_rate_boot_binary_agegroup_dynamic.png")
    )

    forest_plot(
        rows_sz,
        f"Bootstrapped Odds Ratios: Seizure Freq (≥{med_sz:.2f}/mo) ~ Sex + Epilepsy Type + Age Group",
        (0.0, 2.0),
        [0.0, 0.5, 1.0, 1.5, 2.0],
        "", # Bottom axis label is empty
        "← Fewer Seizures",
        "More Seizures →",
        os.path.join(save_dir, "forest_seizure_freq_boot_binary_agegroup_dynamic.png")
    )
    print("Done plotting. All outputs saved to:", save_dir)

if __name__ == "__main__":
    main()