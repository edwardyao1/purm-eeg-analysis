import pandas as pd
import numpy as np
import os
import ast
import warnings
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, gaussian_kde
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

# ==========================================================
# TRACKING & PARSING HELPERS
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
# STATISTICS & PLOTTING HELPERS (MWU)
# ==========================================================
def mwu_effect_size(df, group_col, value_col, g1="M", g2="F"):
    x = df[df[group_col] == g1][value_col].dropna()
    y = df[df[group_col] == g2][value_col].dropna()
    n1, n2 = len(x), len(y)
    
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, n1, n2

    u, p = mannwhitneyu(x, y, alternative="two-sided")
    effect = np.nan if n1 * n2 == 0 else 1 - (2 * u) / (n1 * n2)
    return u, p, effect, n1, n2

def calculate_density_jitter(y_vals, width=0.3):
    if len(y_vals) < 3: return np.zeros(len(y_vals))
    y_log = np.log1p(y_vals)
    y_log_noisy = y_log + np.random.normal(0, 1e-6, size=len(y_log))
    
    try:
        kde = gaussian_kde(y_log_noisy)
        density = kde(y_log_noisy)
        if density.max() > 0:
            density = density / density.max()
        return np.random.uniform(-width, width, size=len(y_vals)) * density
    except Exception:
        return np.random.uniform(-width, width, size=len(y_vals))

def plot_scatter_with_stats(df, target_var, title, ylabel, save_path):
    u, p, es, n1, n2 = mwu_effect_size(df, "nlp_gender", target_var, g1="M", g2="F")
    if n1 == 0 or n2 == 0:
        print(f"Skipping plot for {title} due to insufficient data.")
        return

    plot_df = df.copy()
    plot_df['Sex'] = plot_df['nlp_gender'].map({'F': 'Female', 'M': 'Male'})
    plot_df[target_var] = plot_df[target_var].clip(lower=0) 
    
    plt.figure(figsize=(8, 8))
    sns.set_theme(style="ticks") 
    
    ax = sns.boxplot(
        x="Sex", y=target_var, data=plot_df, order=["Female", "Male"], showfliers=False, width=0.4, 
        boxprops=dict(facecolor="lightgray", edgecolor="black", linewidth=2.5, alpha=0.7),
        medianprops=dict(color="black", linewidth=3.5),
        whiskerprops=dict(color="black", linewidth=2.5),
        capprops=dict(color="black", linewidth=2.5)
    )

    custom_palette = {"Female": "#e24a33", "Male": "#348abd"}
    for i, group in enumerate(["Female", "Male"]):
        group_mask = plot_df['Sex'] == group
        y_vals = plot_df.loc[group_mask, target_var].values
        if len(y_vals) > 0:
            jitter = calculate_density_jitter(y_vals, width=0.25)
            ax.scatter(i + jitter, y_vals, color=custom_palette[group], s=20, alpha=0.6, edgecolors='none', zorder=3)
    
    plt.yscale('symlog', linthresh=0.01)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.7)
    sns.despine()

    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.ylabel(f"{ylabel} (Log Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Sex", fontsize=14, fontweight='bold')
    
    stats_text = f"Mann-Whitney U = {u:.2f}\np-value = {p:.4f}\nEffect Size = {es:.4f}\nn (Female) = {n2}\nn (Male) = {n1}"
    plt.annotate(stats_text, xy=(0.95, 0.05), xycoords='axes fraction', horizontalalignment='right', 
                 verticalalignment='bottom', fontsize=11, bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="lightgray", lw=1, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_stratified_scatter_with_stats(df, target_var, title, ylabel, save_path):
    plot_df = df.copy()
    plot_df['Sex'] = plot_df['nlp_gender'].map({'F': 'Female', 'M': 'Male'})
    plot_df[target_var] = plot_df[target_var].clip(lower=0) 
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    sns.set_theme(style="ticks") 
    custom_palette = {"Female": "#e24a33", "Male": "#348abd"}
    
    for ax, subtype in zip(axes, ["Focal", "General"]):
        sub_df = plot_df[plot_df['canonical_subtype'] == subtype]
        u, p, es, n1, n2 = mwu_effect_size(sub_df, "nlp_gender", target_var, g1="M", g2="F")
        
        sns.boxplot(
            x="Sex", y=target_var, data=sub_df, order=["Female", "Male"], showfliers=False, width=0.4, 
            boxprops=dict(facecolor="lightgray", edgecolor="black", linewidth=2.5, alpha=0.7),
            medianprops=dict(color="black", linewidth=3.5),
            whiskerprops=dict(color="black", linewidth=2.5),
            capprops=dict(color="black", linewidth=2.5), ax=ax
        )

        for i, group in enumerate(["Female", "Male"]):
            group_mask = sub_df['Sex'] == group
            y_vals = sub_df.loc[group_mask, target_var].values
            if len(y_vals) > 0:
                jitter = calculate_density_jitter(y_vals, width=0.25)
                ax.scatter(i + jitter, y_vals, color=custom_palette[group], s=20, alpha=0.6, edgecolors='none', zorder=3)
        
        ax.set_yscale('symlog', linthresh=0.01)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.7)
        ax.set_title(f"{subtype} Epilepsy", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Sex", fontsize=14, fontweight='bold')
        if ax == axes[0]: ax.set_ylabel(f"{ylabel} (Log Scale)", fontsize=14, fontweight='bold')
        
        if n1 > 0 and n2 > 0:
            stats_text = f"Mann-Whitney U = {u:.2f}\np-value = {p:.4f}\nEffect Size = {es:.4f}\nn (Female) = {n2}\nn (Male) = {n1}"
            ax.annotate(stats_text, xy=(0.95, 0.05), xycoords='axes fraction', horizontalalignment='right', 
                        verticalalignment='bottom', fontsize=11, bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="lightgray", lw=1, alpha=0.9))

    sns.despine()
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_statistical_analysis(patient_df, target_var, target_name):
    print(f"\n=== MANN-WHITNEY U: {target_name.upper()} ===")
    u, p, es, n1, n2 = mwu_effect_size(patient_df, "nlp_gender", target_var)
    print(f"Overall -> M = {n1}, F = {n2} | U = {u:.2f}, p = {p:.6f}, effect = {es:.4f}")

    type_pvals, type_results = [], []
    for t in ["Focal", "General"]:
        sub = patient_df[patient_df["canonical_subtype"] == t]
        u, p, es, n1, n2 = mwu_effect_size(sub, "nlp_gender", target_var)
        type_results.append((t, u, p, es, n1, n2))
        type_pvals.append(1.0 if pd.isna(p) else p)

    reject, p_adj, _, _ = multipletests(type_pvals, method="bonferroni")
    for (t, u, p, es, n1, n2), pa, sig in zip(type_results, p_adj, reject):
        print(f"Subtype: {t} -> M = {n1}, F = {n2} | U = {u:.2f}, Raw p = {p:.6f}, Bonf p = {pa:.6f}, Effect = {es:.4f}, Sig = {sig}")

# ==========================================================
# BOOTSTRAP & OLS HELPERS
# ==========================================================
def bootstrap_regression_coeffs(df, outcome_col, formula_vars, n_boot=5000):
    df = df.copy()

    # remove impossible values
    df = df[np.isfinite(df[outcome_col])]

    if outcome_col == "mean_sz_freq":
        df = df[df[outcome_col] >= 0]

    df[outcome_col] = np.log1p(df[outcome_col])

    # Adjusted dummy mapping to match 'canonical_subtype' from the first script
    reg_df = pd.get_dummies(
        df[[outcome_col, "nlp_gender", "canonical_subtype"]],
        drop_first=True
    ).astype(float)

    results = {}
    for var in formula_vars:
        if var not in reg_df.columns:
            results[var] = (0, 0, 0, 1)
            continue

        coeffs = []
        for _ in range(n_boot):
            sample = reg_df.sample(len(reg_df), replace=True)
            X = sample[formula_vars]
            X = sm.add_constant(X)
            y = sample[outcome_col]

            try:
                model = sm.OLS(y, X).fit()
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

def run_bootstrap_analysis(patient_df, outcome_col, outcome_label):
    print(f"\n===========================================================================")
    print(f"BOOTSTRAP REGRESSION — {outcome_label}")
    print(f"===========================================================================")
    
    # Updated to match dummy structure of 'canonical_subtype'
    formula_vars = ['nlp_gender_M', 'canonical_subtype_General']
    coeffs = bootstrap_regression_coeffs(patient_df, outcome_col, formula_vars)
    
    rows = []
    
    # Section 1: Sex
    m_res = coeffs['nlp_gender_M']
    female_n = len(patient_df[patient_df['nlp_gender']=='F'])
    rows.append(dict(section="Sex", label="Female", n=female_n, ref_med=np.median(patient_df[patient_df['nlp_gender']=='F'][outcome_col]), is_reference=True, diff=0, lo=0, hi=0, p=None))
    rows.append(dict(section="Sex", label="Male", n=len(patient_df[patient_df['nlp_gender']=='M']), ref_med=None, is_reference=False, diff=m_res[0], lo=m_res[1], hi=m_res[2], p=m_res[3]))

    # Section 2: Epilepsy
    g_res = coeffs['canonical_subtype_General']
    focal_n = len(patient_df[patient_df['canonical_subtype']=='Focal'])
    rows.append(dict(section="Epilepsy Type", label="Focal", n=focal_n, ref_med=np.median(patient_df[patient_df['canonical_subtype']=='Focal'][outcome_col]), is_reference=True, diff=0, lo=0, hi=0, p=None))
    rows.append(dict(section="Epilepsy Type", label="General", n=len(patient_df[patient_df['canonical_subtype']=='General']), ref_med=None, is_reference=False, diff=g_res[0], lo=g_res[1], hi=g_res[2], p=g_res[3]))

    return rows

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

def print_ols_results(df, outcome_col, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    reg_df = df.copy()
    reg_df = reg_df[["nlp_gender", "canonical_subtype", outcome_col]].copy()

    # force outcome numeric
    reg_df[outcome_col] = pd.to_numeric(reg_df[outcome_col], errors="coerce")
    reg_df = reg_df.dropna(subset=[outcome_col])
    reg_df = reg_df[np.isfinite(reg_df[outcome_col])].copy()

    if outcome_col == "mean_sz_freq":
        reg_df = reg_df[reg_df[outcome_col] >= 0].copy()

    # log transform
    reg_df[outcome_col] = np.log1p(reg_df[outcome_col])

    reg_df["nlp_gender_M"] = (reg_df["nlp_gender"] == "M").astype(float)
    reg_df["canonical_subtype_General"] = (reg_df["canonical_subtype"] == "General").astype(float)

    X = reg_df[["nlp_gender_M", "canonical_subtype_General"]].astype(float)
    X = sm.add_constant(X)
    y = reg_df[outcome_col].astype(float)

    model = sm.OLS(y, X).fit()
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

    clinical_df = pd.read_csv(clinical_csv, low_memory=False).rename(columns={'patient_id': 'Patient', 'session_number': 'Session'})
    spike_df = pd.read_csv(spike_csv, low_memory=False)

    print("======================================================")
    print("           DATA CLEANING & COHORT ATTRITION           ")
    print("======================================================\n")

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
        
        # Put this right after you filter patient_sz_freq in Python to find the extra 15
        print("Python kept these patient IDs at this step:", patient_sz_freq['Patient'].tolist())
    else:
        current_patients = set()

    track_patients("Base Filter: Documented Seizure Frequency (Primary Cohort)", before_pats, current_patients)

    # =====================================================================
    # DEBUG INJECTION: Save Python's intermediate cohort right after Sz Freq
    # =====================================================================
    debug_df = pd.DataFrame({'Patient_ID': list(current_patients)})
    debug_out_path = os.path.join(save_dir, 'python_kept_after_sz_freq.csv')
    debug_df.to_csv(debug_out_path, index=False)
    print(f"DEBUG: Saved {len(current_patients)} patient IDs to {debug_out_path}\n")
    # =====================================================================

    # 4. CALCULATE PATIENT SPIKE RATES
    patient_spikes = valid_sessions[valid_sessions['Patient'].isin(current_patients)].groupby('Patient').agg(
        total_spikes=("count_0_46", "sum"),
        total_duration=("Duration_sec", "sum")
    ).reset_index()
    patient_spikes["spike_rate_per_hour"] = (patient_spikes["total_spikes"] / patient_spikes["total_duration"]) * 3600

    # 5. PROJECT SPECIFIC FILTERS
    print("======================================================")
    print("               PROJECT SPECIFIC FILTERS               ")
    print("======================================================\n")
    
    final_df = pd.merge(patient_sz_freq, patient_spikes, on='Patient', how='inner')
    final_df = final_df.merge(pt_demo[['nlp_gender', 'epilepsy_type', 'epilepsy_specific']].reset_index(), on='Patient', how='inner')
    
    before_pats = current_patients.copy()
    final_df = final_df[final_df['nlp_gender'].isin(['M', 'F'])]
    current_patients = set(final_df['Patient'].unique())
    track_patients("Project Filter A: Valid Gender (M or F)", before_pats, current_patients)

    before_pats = current_patients.copy()
    final_df['canonical_subtype'] = final_df.apply(assign_canonical_subtype, axis=1)
    final_df = final_df[final_df['canonical_subtype'].isin(['Focal', 'General'])]
    current_patients = set(final_df['Patient'].unique())
    track_patients("Project Filter B: Focal or General Subtype", before_pats, current_patients)

    print("======================================================")
    print(f"FINAL COHORT SIZE FOR ANALYSIS: {len(current_patients)} patients")
    print("======================================================\n")
    
   # =====================================================================
    # 5.5 CALCULATE BASELINE PATIENT AGES IN FINAL COHORT
    # =====================================================================
    print("======================================================")
    print("                 VALID AGE CALCULATION                ")
    print("======================================================\n")
    
    dob_col = 'deid_birth_date' 
    
    if dob_col in clinical_df.columns:
        age_df = clinical_df[clinical_df['Patient'].isin(current_patients)][['Patient', dob_col]].drop_duplicates().copy()
        
        # 1. Anchor Date: First Clinic Visit is defined as Jan 1, 2000
        age_df['anchor_date'] = pd.to_datetime('2000-01-01')
        
        # 2. Parse Birth Dates (letting pandas figure out the format since it's already YYYY-MM-DD)
        # Using mixed format handles both 'YYYY-MM-DD' and any lingering 'M/D/Y'
        age_df['dob_dt'] = pd.to_datetime(age_df[dob_col], format='mixed', errors='coerce')
        
        # Fix the 2000s vs 1900s issue for any 2-digit years
        def fix_century(dt):
            if pd.isna(dt): return dt
            if dt.year > 2025: return dt.replace(year=dt.year - 100)
            return dt
            
        age_df['dob_dt'] = age_df['dob_dt'].apply(fix_century)
        
        # 3. Calculate Age in Years at the first visit
        age_df['baseline_age'] = (age_df['anchor_date'] - age_df['dob_dt']).dt.days / 365.25
        
        # Filter for "valid" ages (0 to 120 years old)
        valid_age_mask = (age_df['baseline_age'] >= 0) & (age_df['baseline_age'] <= 120)
        valid_age_df = age_df[valid_age_mask]
        
        # Count unique patients
        patients_with_valid_age = valid_age_df['Patient'].nunique()
        total_final_patients = len(current_patients)
        
        print(f"Total patients in final cohort: {total_final_patients}")
        print(f"Patients with valid baseline age: {patients_with_valid_age}")
        print(f"Patients missing valid ages: {total_final_patients - patients_with_valid_age}\n")
        
        # Merge baseline age into final_df
        if patients_with_valid_age > 0:
            final_df = final_df.merge(valid_age_df[['Patient', 'baseline_age']], on='Patient', how='inner')
            # The 'inner' merge will automatically drop patients without a valid age from the final analysis
            current_patients = set(final_df['Patient'].unique())
            track_patients("Project Filter C: Valid Age Calculated", set(age_df['Patient'].unique()), current_patients)
        
    else:
        print(f"WARNING: '{dob_col}' column not found. Cannot calculate valid ages.\n")
        
    # --- Identify and Print the 3 Missing Patients ---
    invalid_age_mask = ~age_df['Patient'].isin(valid_age_df['Patient'])
    missing_patients_df = age_df[invalid_age_mask]
        
    print("=== DETAILS OF PATIENTS WITH INVALID AGES ===")
    print(missing_patients_df[['Patient', dob_col, 'dob_dt', 'baseline_age']])
    print("=============================================\n")
    
    # =====================================================================
    # --- Save Cohort to Spreadsheet ---
    # =====================================================================
    output_df = pd.DataFrame({'Patient_ID': list(current_patients)})
    
    # Save it to the same output directory you defined earlier
    python_out_path = os.path.join(save_dir, 'python_kept_patients.csv')
    output_df.to_csv(python_out_path, index=False)
    print(f"Saved patient list to: {python_out_path}\n")
    # =====================================================================

    # 6. RUN ANALYSIS & SCATTER PLOTS (Original Pipeline)
    run_statistical_analysis(final_df, target_var="spike_rate_per_hour", target_name="Spike Rate (per hour)")
    run_statistical_analysis(final_df, target_var="mean_sz_freq", target_name="Average Seizure Frequency")

    print("\n--- Generating Scatter Plots ---")
    plot_scatter_with_stats(final_df, "spike_rate_per_hour", "Figure 1\nOverall Men vs Women: Spike Rate", "Spikes per Hour", os.path.join(save_dir, "TESTING_Figure1_Overall_Spike_Rate.png"))
    plot_scatter_with_stats(final_df, "mean_sz_freq", "Figure 2\nOverall Men vs Women: Seizure Frequency", "Average Seizure Frequency", os.path.join(save_dir, "TESTING_Figure2_Overall_Seizure_Frequency.png"))
    plot_stratified_scatter_with_stats(final_df, "spike_rate_per_hour", "Figure 3: Spike Rate by Sex (Focal vs General)", "Spikes per Hour", os.path.join(save_dir, "TESTING_Figure3_Spike_Rate_Stratified.png"))
    plot_stratified_scatter_with_stats(final_df, "mean_sz_freq", "Figure 4: Seizure Frequency by Sex (Focal vs General)", "Average Seizure Frequency", os.path.join(save_dir, "TESTING_Figure4_Seizure_Frequency_Stratified.png"))

    # 7. RUN STANDARD OLS REGRESSION
    print_ols_results(final_df, "spike_rate_per_hour", "OLS REGRESSION: SPIKE RATE")
    print_ols_results(final_df, "mean_sz_freq", "OLS REGRESSION: SEIZURE FREQUENCY")

    # 8. RUN BOOTSTRAP OLS REGRESSION & FOREST PLOTS
    rows_spike = run_bootstrap_analysis(final_df, "spike_rate_per_hour", "SPIKE RATE (per hour)")
    rows_sz = run_bootstrap_analysis(final_df, "mean_sz_freq", "SEIZURE FREQUENCY (per month)")
    
    print("\n--- Generating Forest Plots ---")
    forest_plot(
        rows_spike,
        "Bootstrapped OLS Coefficients: Spike Rate (per hour)",
        (-0.2, 0.8),
        [-0.2, 0, 0.2, 0.4, 0.6, 0.8],
        "Coefficient", 
        os.path.join(save_dir, "TESTING_forest_spike_rate_boot_ols.png")
    )
    
    forest_plot(
        rows_sz,
        "Bootstrapped OLS Coefficients: Seizure Frequency (per month)",
        (-0.2, 0.5),
        [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "Coefficient", 
        os.path.join(save_dir, "TESTING_forest_seizure_freq_boot_ols.png")
    )

    print("\nPipeline Complete. All outputs saved to:", save_dir)

if __name__ == "__main__":
    main()