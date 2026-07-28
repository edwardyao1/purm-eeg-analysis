import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, gaussian_kde
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import os

def track_filter(step_name, before, after):
    removed = before - after
    print(f"--- {step_name} ---")
    print(f"Started with {before} rows, removed {removed}. Remaining: {after}\n")

# ==========================================================
# PARSERS & HELPERS
# ==========================================================
def parse_sz_freq(val):
    """
    Safely converts sz_freqs strings like '[2.5]' or '[1.0, null, 2.0]' into a single numeric float.
    Takes the mean if multiple valid numbers are present in the list.
    """
    if pd.isna(val): 
        return np.nan
    
    val_str = str(val).strip().strip("[]").replace("'", "").replace('"', "")
    if not val_str: 
        return np.nan
    
    try:
        parts = [float(x.strip()) for x in val_str.split(",") if x.strip() and x.strip().lower() != 'null']
        if not parts:
            return np.nan
        return np.mean(parts)
    except ValueError:
        return np.nan

def mwu_effect_size(df, group_col, value_col, g1="M", g2="F"):
    """
    Performs a Mann-Whitney U test between two groups and computes an effect size.
    """
    x = df[df[group_col] == g1][value_col].dropna()
    y = df[df[group_col] == g2][value_col].dropna()

    n1, n2 = len(x), len(y)

    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, n1, n2

    u, p = mannwhitneyu(x, y, alternative="two-sided")
    
    if n1 * n2 == 0:
        effect = np.nan
    else:
        effect = 1 - (2 * u) / (n1 * n2)

    return u, p, effect, n1, n2

def canonical_subtype(row):
    etype = str(row.get("epilepsy_type", "")).lower()
    espec = str(row.get("epilepsy_specific", "")).lower()

    if "temporal" in espec: return "Temporal"
    if "frontal" in espec: return "Frontal"
    if "focal" in etype: return "Focal"
    if etype == "general": return "General"
    
    return np.nan

def calculate_density_jitter(y_vals, width=0.3):
    """
    Calculates a custom horizontal jitter based on data density.
    This creates the "bulge" shape where there are more dots, while allowing overlap.
    """
    if len(y_vals) < 3:
        return np.zeros(len(y_vals))
        
    # Log-transform helps the KDE correctly estimate density for highly skewed data
    y_log = np.log1p(y_vals)
    # Microscopic noise prevents crashes if there are too many identical values (like exact 0s)
    y_log_noisy = y_log + np.random.normal(0, 1e-6, size=len(y_log))
    
    try:
        kde = gaussian_kde(y_log_noisy)
        density = kde(y_log_noisy)
        # Normalize so the widest section equals our maximum 'width'
        if density.max() > 0:
            density = density / density.max()
        
        # Jitter randomly within the bounds of the calculated density
        jitter = np.random.uniform(-width, width, size=len(y_vals)) * density
        return jitter
    except Exception:
        # Fallback to standard uniform jitter if KDE fails
        return np.random.uniform(-width, width, size=len(y_vals))

# ==========================================================
# PLOTTING FUNCTION
# ==========================================================
def plot_scatter_with_stats(df, target_var, title, ylabel, save_path):
    """
    Generates a custom density-scatter plot allowing overlap, over a shaded boxplot.
    """
    # FILTER: Remove rows where the target variable is 0
    df = df[df[target_var] > 0].copy()

    # 1. Run Stats
    u, p, es, n1, n2 = mwu_effect_size(df, "nlp_gender", target_var, g1="M", g2="F")
    
    if n1 == 0 or n2 == 0:
        print(f"Skipping plot for {title} due to insufficient data.")
        return

    # 2. Prep Data for Plotting
    plot_df = df.copy()
    plot_df['Sex'] = plot_df['nlp_gender'].map({'F': 'Female', 'M': 'Male'})
    
    # 3. Setup Plot Aesthetics
    plt.figure(figsize=(8, 8))
    sns.set_theme(style="ticks") 
    
    # 4. Create Prominent Shaded Boxplot FIRST (Background)
    ax = sns.boxplot(
        x="Sex", y=target_var, data=plot_df, 
        order=["Female", "Male"], 
        showfliers=False, 
        width=0.4, 
        boxprops=dict(facecolor="lightgray", edgecolor="black", linewidth=2.5, alpha=0.7),
        medianprops=dict(color="black", linewidth=3.5),
        whiskerprops=dict(color="black", linewidth=2.5),
        capprops=dict(color="black", linewidth=2.5)
    )

    # 5. Create Custom Density-Spread Dots OVERLAY (Foreground)
    custom_palette = {"Female": "#e24a33", "Male": "#348abd"}
    
    for i, group in enumerate(["Female", "Male"]):
        group_mask = plot_df['Sex'] == group
        y_vals = plot_df.loc[group_mask, target_var].values
        
        if len(y_vals) > 0:
            # Apply our custom density jitter
            jitter = calculate_density_jitter(y_vals, width=0.25)
            x_vals = i + jitter
            
            # Draw standard scatter points (allows overlap seamlessly)
            ax.scatter(
                x_vals, y_vals, 
                color=custom_palette[group], 
                s=20,          # Dot size
                alpha=0.6,     # Transparency allows you to see density where dots overlap
                edgecolors='none',
                zorder=3       # Forces dots to sit visually on top of the boxplot
            )
    
    # 6. Apply Log Scale & Axis Formatting
    plt.yscale('symlog', linthresh=0.01)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.7)
    sns.despine()

    # 7. Labels
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.ylabel(f"{ylabel} (Log Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Sex", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 8. Annotate Statistics
    stats_text = (f"Mann-Whitney U = {u:.2f}\n"
                  f"p-value = {p:.4f}\n"
                  f"Effect Size = {es:.4f}\n"
                  f"n (Female) = {n2}\n"
                  f"n (Male) = {n1}")
    
    plt.annotate(
        stats_text, xy=(0.95, 0.05), xycoords='axes fraction', 
        horizontalalignment='right', verticalalignment='bottom', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="lightgray", lw=1, alpha=0.9)
    )
    
    # 9. Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved styled plot to: {save_path}")

# ==========================================================
# REUSABLE ANALYSIS FUNCTION
# ==========================================================
def run_statistical_analysis(patient_df, target_var, target_name):
    # FILTER: Remove rows where the target variable is 0
    patient_df = patient_df[patient_df[target_var] > 0].copy()

    print(f"\n======================================================")
    print(f"=== MANN-WHITNEY U: {target_name.upper()} (>0 ONLY) ===")
    print(f"======================================================\n")

    # 1. PRIMARY COHORT
    print("=== PRIMARY: OVERALL COHORT ===")
    u, p, es, n1, n2 = mwu_effect_size(patient_df, "nlp_gender", target_var)
    print(f"M = {n1}, F = {n2}")
    if n1 > 0 and n2 > 0:
        print(f"U = {u:.2f}, p = {p:.6f}, effect = {es:.4f}")
    else:
        print("Insufficient data for analysis.")

    # 2. EPILEPSY TYPE
    print("\n=== EPILEPSY TYPE ANALYSIS ===")
    type_results = []
    type_pvals = []

    for t in ["Focal", "General"]:
        sub = patient_df[patient_df["epilepsy_type"] == t]
        u, p, es, n1, n2 = mwu_effect_size(sub, "nlp_gender", target_var)
        type_results.append((t, u, p, es, n1, n2))
        type_pvals.append(1.0 if pd.isna(p) else p)

    reject, p_adj, _, _ = multipletests(type_pvals, method="bonferroni")

    for (t, u, p, es, n1, n2), pa, sig in zip(type_results, p_adj, reject):
        print(f"\nType: {t}")
        print(f"M = {n1}, F = {n2}")
        if n1 == 0 or n2 == 0:
            print("Insufficient data.")
        else:
            print(f"U = {u:.2f} | Raw p = {p:.6f} | Bonf p = {pa:.6f} | Effect = {es:.4f} | Sig = {sig}")

    # 3. FOCAL SUBTYPES
    print("\n=== FOCAL SUBTYPES ===")
    focal_df = patient_df[patient_df["epilepsy_type"] == "Focal"].copy()
    focal_df = focal_df[focal_df["canonical_subtype"].isin(["Temporal", "Frontal"])]
    
    focal_results = []
    focal_pvals = []

    for st in ["Temporal", "Frontal"]:
        sub = focal_df[focal_df["canonical_subtype"] == st]
        u, p, es, n1, n2 = mwu_effect_size(sub, "nlp_gender", target_var)
        focal_results.append((st, u, p, es, n1, n2))
        focal_pvals.append(1.0 if pd.isna(p) else p)

    reject, p_adj, _, _ = multipletests(focal_pvals, method="bonferroni")

    for (st, u, p, es, n1, n2), pa, sig in zip(focal_results, p_adj, reject):
        print(f"\nSubtype: {st}")
        print(f"M = {n1}, F = {n2}")
        if n1 == 0 or n2 == 0:
            print("Insufficient data.")
        else:
            print(f"U = {u:.2f} | Raw p = {p:.6f} | Bonf p = {pa:.6f} | Effect = {es:.4f} | Sig = {sig}")

# ==========================================================
# MAIN
# ==========================================================
def main():
    # ----------------------------
    # DIRECTORIES & PATHS
    # ----------------------------
    data_dir = '/Users/edwardyao/Documents/PURM/data/'
    save_dir = '/Users/edwardyao/Documents/PURM/gender_project_output/'
    clinical_csv = os.path.join(data_dir, 'clinical_data_deidentified.csv')
    spike_csv = os.path.join(data_dir, 'spike_counts.csv')

    # ----------------------------
    # LOAD
    # ----------------------------
    clinical_df = pd.read_csv(clinical_csv)
    spike_df = pd.read_csv(spike_csv)

    # ----------------------------
    # 1. INITIAL MERGE (OUTER JOIN)
    # ----------------------------
    clinical_df = clinical_df.rename(columns={'patient_id': 'Patient', 'session_number': 'Session'})
    df = pd.merge(spike_df, clinical_df, on=["Patient", "Session"], how="outer")
    print(f"--- Initial Merge ---\nMerged dataset starts with {len(df)} total sessions\n")

    # ----------------------------
    # 2. APPLY SESSION FILTERS
    # ----------------------------
    allowable_visits = {
        "CONSULT VISIT", "ESTABLISHED PATIENT VISIT", "FOLLOW-UP PATIENT CLINIC",
        "NEW PATIENT CLINIC", "NEW PATIENT VISIT", "NPV MANAGEMENT DURING COVID-19",
        "NPV NEUROLOGY", "RETURN ANNUAL VISIT", "RETURN PATIENT EXTENDED",
        "RETURN PATIENT VISIT", "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN"
    }

    before = len(df)
    df = df[df["visit_type"].astype(str).apply(lambda v: any(x in v for x in allowable_visits))].copy()
    track_filter("Filter: Allowable Visits", before, len(df))

    acq = df["acquired_on"].fillna("").astype(str).str.lower()
    patient_class = df["report_PATIENT_CLASS"].fillna("").astype(str).str.lower()
    jay = df["jay_in_or_out"].fillna("").astype(str).str.lower()
    
    before = len(df)
    df = df[acq.str.contains("spe") | acq.str.contains("radnor") | (patient_class == "outpatient") | (jay == "out")].copy()
    track_filter("Filter: MATLAB Acquisition Locations", before, len(df))

    before = len(df)
    df = df[df["Duration_sec"] > 0].copy()
    track_filter("Filter: Duration > 0", before, len(df))

    before = len(df)
    df = df[df["Duration_sec"] <= 4 * 3600].copy()
    track_filter("Filter: Duration <= 4 hours", before, len(df))

    before = len(df)
    df = df[df["count_0_46"].notna()].copy()
    track_filter("Filter: Valid count_0_46 (Spikes)", before, len(df))
    
    before = len(df)
    df = df[df["nlp_gender"].isin(["M", "F"])].copy()
    track_filter("Filter: Valid Gender (M/F)", before, len(df))
    
    bad_types = {"Uncertain if Epilepsy", "Unknown or MRN not found", "", "Non-Epileptic Seizure Disorder", "Unclassified or Unspecified"}
    before = len(df)
    df = df[~df["epilepsy_type"].isin(bad_types)].copy()
    track_filter("Filter: Remove Bad Epilepsy Types", before, len(df))

    before = len(df)
    df["sz_freq_numeric"] = df["sz_freqs"].apply(parse_sz_freq)
    df = df[df["sz_freq_numeric"].notna()].copy()
    track_filter("Filter: Valid Numeric Seizure Frequency", before, len(df))

    df["canonical_subtype"] = df.apply(canonical_subtype, axis=1)

    # ----------------------------
    # 3. AGGREGATE TO PATIENT LEVEL
    # ----------------------------
    patient_df = df.groupby(
        ["Patient", "nlp_gender", "epilepsy_type", "canonical_subtype"], 
        dropna=False
    ).agg(
        total_spikes=("count_0_46", "sum"),
        total_duration=("Duration_sec", "sum"),
        mean_sz_freq=("sz_freq_numeric", "mean") 
    ).reset_index()

    patient_df["spike_rate_per_hour"] = (patient_df["total_spikes"] / patient_df["total_duration"]) * 3600

    before_drop = len(patient_df)
    patient_df = patient_df.dropna(subset=["spike_rate_per_hour", "mean_sz_freq"])

    print("--- Aggregating to Patient Level ---")
    print(f"Grouped {len(df)} valid sessions into {before_drop} patients.")
    print(f"Removed {before_drop - len(patient_df)} patients missing final aggregated metrics.")
    print(f"FINAL PATIENT COHORT SIZE: {len(patient_df)}\n")

    # ----------------------------
    # 4. RUN STATISTICS
    # ----------------------------
    run_statistical_analysis(patient_df, target_var="spike_rate_per_hour", target_name="Spike Rate (per hour)")
    run_statistical_analysis(patient_df, target_var="mean_sz_freq", target_name="Average Seizure Frequency")

    # ----------------------------
    # 5. GENERATE & SAVE PLOTS
    # ----------------------------
    print("\n--- Generating Plots ---")
    
    spike_save_path = os.path.join(save_dir, "A_Spike_Rate_Scatter_Without_Zeroes.png")
    plot_scatter_with_stats(
        df=patient_df, 
        target_var="spike_rate_per_hour", 
        title="A\nOverall Men vs Women: Spike Rate (>0)", 
        ylabel="Spikes per Hour", 
        save_path=spike_save_path
    )
    
    sz_save_path = os.path.join(save_dir, "B_Seizure_Frequency_Scatter_Without_Zeroes.png")
    plot_scatter_with_stats(
        df=patient_df, 
        target_var="mean_sz_freq", 
        title="B\nOverall Men vs Women: Seizure Frequency (>0)", 
        ylabel="Average Seizure Frequency", 
        save_path=sz_save_path
    )

if __name__ == "__main__":
    main()