import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

# ==========================================================
# UTILITIES & FILTERING PIPELINE
# ==========================================================
def canonical_subtype(row):
    etype = str(row.get("epilepsy_type", "")).lower()
    espec = str(row.get("epilepsy_specific", "")).lower()

    # Added Frontal and Temporal here (matching your existing logic)
    if "frontal" in espec: return "Frontal"
    if "temporal" in espec: return "Temporal"
    if "focal" in etype: return "Focal"
    if etype == "general": return "General"
    return np.nan

def load_and_filter_data():
    clinical_df = pd.read_csv('/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv')
    spike_df = pd.read_csv('/Users/edwardyao/Documents/PURM/data/spike_counts.csv')

    clinical_df = clinical_df.rename(columns={'patient_id': 'Patient', 'session_number': 'Session'})
    df = pd.merge(spike_df, clinical_df, on=["Patient", "Session"], how="outer")

    allowable_visits = {
        "CONSULT VISIT", "ESTABLISHED PATIENT VISIT", "FOLLOW-UP PATIENT CLINIC",
        "NEW PATIENT CLINIC", "NEW PATIENT VISIT", "NPV MANAGEMENT DURING COVID-19",
        "NPV NEUROLOGY", "RETURN ANNUAL VISIT", "RETURN PATIENT EXTENDED",
        "RETURN PATIENT VISIT", "RPV MANAGEMENT DURING COVID-19", "TELEHEALTH VIDEO VISIT RETURN",
    }
    
    df = df[df["visit_type"].astype(str).apply(lambda v: any(x in v for x in allowable_visits))].copy()

    acq = df["acquired_on"].fillna("").astype(str).str.lower()
    patient_class = df["report_PATIENT_CLASS"].fillna("").astype(str).str.lower()
    jay = df["jay_in_or_out"].fillna("").astype(str).str.lower()
    
    df = df[acq.str.contains("spe") | acq.str.contains("radnor") | (patient_class == "outpatient") | (jay == "out")].copy()
    df = df[df["Duration_sec"] > 0].copy()
    df = df[df["Duration_sec"] <= 4 * 3600].copy()
    df = df[df["count_0_46"].notna()].copy()
    df = df[df["nlp_gender"].isin(["M", "F"])].copy()
    
    bad_types = {"Uncertain if Epilepsy", "Unknown or MRN not found", "", "Non-Epileptic Seizure Disorder", "Unclassified or Unspecified"}
    df = df[~df["epilepsy_type"].isin(bad_types)].copy()
    df["canonical_subtype"] = df.apply(canonical_subtype, axis=1)

    patient_df = df.groupby(["Patient", "nlp_gender", "canonical_subtype"], dropna=False).agg(
        total_spikes=("count_0_46", "sum"),
        total_duration=("Duration_sec", "sum")
    ).reset_index()

    patient_df = patient_df[patient_df['total_duration'] > 0].copy()
    patient_df["spike_rate"] = (patient_df["total_spikes"] / patient_df["total_duration"] * 3600)
    patient_df['canonical_subtype'] = patient_df['canonical_subtype'].replace({'General': 'Generalized'})
    
    return patient_df.dropna(subset=['spike_rate', 'nlp_gender'])


# ==========================================================
# CHUNKED BOOTSTRAPPING (Fast AND Memory-Safe)
# ==========================================================
def bootstrap_median(group, iterations=5000, chunk_size=500):
    np.random.seed(42) 
    n = len(group)
    if n == 0: return np.nan, np.nan, np.nan
    
    group = np.asarray(group)
    medians = np.empty(iterations)
    
    for i in range(0, iterations, chunk_size):
        end = min(i + chunk_size, iterations)
        current_chunk = end - i
        idx = np.random.randint(0, n, size=(current_chunk, n))
        medians[i:end] = np.median(group[idx], axis=1)
        
    ci_lower = np.percentile(medians, 2.5)
    ci_upper = np.percentile(medians, 97.5)
    return np.median(group), ci_lower, ci_upper


def bootstrap_median_diff(group1, group2, iterations=5000, chunk_size=500):
    np.random.seed(42) 
    n1, n2 = len(group1), len(group2)
    group1, group2 = np.asarray(group1), np.asarray(group2)
    
    diffs = np.empty(iterations)
    null_diffs = np.empty(iterations)
    
    combined = np.concatenate([group1, group2])
    n_combined = len(combined)
    
    for i in range(0, iterations, chunk_size):
        end = min(i + chunk_size, iterations)
        chunk = end - i
        
        idx1 = np.random.randint(0, n1, size=(chunk, n1))
        idx2 = np.random.randint(0, n2, size=(chunk, n2))
        diffs[i:end] = np.median(group1[idx1], axis=1) - np.median(group2[idx2], axis=1)
        
        null_idx1 = np.random.randint(0, n_combined, size=(chunk, n1))
        null_idx2 = np.random.randint(0, n_combined, size=(chunk, n2))
        null_diffs[i:end] = np.median(combined[null_idx1], axis=1) - np.median(combined[null_idx2], axis=1)
        
    observed_diff = np.median(group1) - np.median(group2)
    ci_lower, ci_upper = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    
    return observed_diff, ci_lower, ci_upper, p_value


# ==========================================================
# FOREST PLOT VISUALIZATION
# ==========================================================
def plot_absolute_medians(results, save_dir):
    """GRAPH 1: Plots absolute Medians for the subgroups."""
    # Increased height to 7 to accommodate new rows
    fig = plt.figure(figsize=(10, 7), facecolor='white')
    fig.suptitle("Predicted Median Spike Rates by Subgroup", fontsize=15, fontweight='bold', y=0.98)
    
    # Widened left column slightly to fit 'n=XXX'
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.2, 2.0, 1.0])
    
    ax_left = fig.add_subplot(gs[0])
    ax_plot = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2])

    n_items = len(results)
    y_pos = np.arange(n_items)[::-1]

    # FORCE EXACT SAME Y-LIMITS ON ALL AXES TO PREVENT MISALIGNMENT
    for ax in [ax_left, ax_plot, ax_right]:
        ax.set_ylim(-1, n_items)

    # LEFT PANEL
    ax_left.set_xlim(0, 1)
    ax_left.axis('off')
    ax_left.text(0.0, n_items - 0.2, "Subgroup", fontweight='bold', va='bottom', ha='left', fontsize=11)
    
    for i, res in enumerate(results):
        y = y_pos[i]
        label = f"{res['Group']} (n={res['n']})"
        indent = 0.05 if " w/ " in res['Group'] or res['Group'].startswith("Men") or res['Group'].startswith("Women") else 0.0
        if "Everyone" in res['Group']: indent = 0.0
        ax_left.text(indent, y, label, va='center', ha='left', fontsize=11)

    # CENTER PANEL
    medians = [r['median'] for r in results]
    err_low = [r['median'] - r['ci_l'] for r in results]
    err_high = [r['ci_u'] - r['median'] for r in results]
    
    ax_plot.errorbar(
        x=medians, y=y_pos, 
        xerr=[err_low, err_high], 
        fmt='s', color='#4C72B0', markersize=7, capsize=0, linewidth=2
    )
    
    ax_plot.get_yaxis().set_visible(False)
    ax_plot.spines['left'].set_visible(False)
    ax_plot.spines['top'].set_visible(False)
    ax_plot.spines['right'].set_visible(False)
    ax_plot.spines['bottom'].set_linewidth(1.5)
    ax_plot.tick_params(axis='x', bottom=True, labelbottom=True, labelsize=10, width=1.5, length=5)
    ax_plot.set_xlabel("Predicted Median Spike Rate (Spikes/hour)", fontsize=11, labelpad=15, fontweight='bold')
    ax_plot.grid(True, axis='x', linestyle='--', alpha=0.5)

    # RIGHT PANEL
    ax_right.set_xlim(0, 1)
    ax_right.axis('off')
    ax_right.text(1.0, n_items - 0.2, "Median (95% CI)", fontweight='bold', va='bottom', ha='right', fontsize=11)
    
    for i, res in enumerate(results):
        y = y_pos[i]
        ci_str = f"{res['median']:.3f} ({res['ci_l']:.3f} to {res['ci_u']:.3f})"
        ax_right.text(1.0, y, ci_str, va='center', ha='right', fontsize=11)

    plt.subplots_adjust(wspace=0.05) 
    
    save_path = f"{save_dir}/graph1_median_rates.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"-> Successfully saved Graph 1 to: {save_path}")


def plot_median_differences(results, save_dir):
    """GRAPH 2: Plots Median Differences and P-Values."""
    # Increased height to fit new categories
    fig = plt.figure(figsize=(10, 4.5), facecolor='white')
    fig.suptitle("Median Differences (Female vs. Male)", fontsize=15, fontweight='bold', y=1.02)
    
    # Widened left column to account for the (F: X, M: Y) strings
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1.8, 0.6])
    
    ax_left = fig.add_subplot(gs[0])
    ax_plot = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2])

    n_items = len(results)
    y_pos = np.arange(n_items)[::-1]

    # FORCE EXACT SAME Y-LIMITS ON ALL AXES TO PREVENT MISALIGNMENT
    for ax in [ax_left, ax_plot, ax_right]:
        ax.set_ylim(-1, n_items)

    # LEFT PANEL
    ax_left.set_xlim(0, 1)
    ax_left.axis('off')
    
    ax_left.text(0.0, n_items - 0.2, "Group", fontweight='bold', va='bottom', ha='left', fontsize=11)
    ax_left.text(1.0, n_items - 0.2, "Diff (95% CI)", fontweight='bold', va='bottom', ha='right', fontsize=11)
    
    for i, res in enumerate(results):
        y = y_pos[i]
        # Append Female & Male sample sizes
        group_label = f"{res['Group']} (F={res['n_f']}, M={res['n_m']})"
        ax_left.text(0.0, y, group_label, va='center', ha='left', fontsize=11)
        ci_str = f"{res['diff']:.2f} ({res['ci_l']:.2f} to {res['ci_u']:.2f})"
        ax_left.text(1.0, y, ci_str, va='center', ha='right', fontsize=11)

    # CENTER PANEL
    ax_plot.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    diffs = [r['diff'] for r in results]
    err_low = [r['diff'] - r['ci_l'] for r in results]
    err_high = [r['ci_u'] - r['diff'] for r in results]
    
    ax_plot.errorbar(
        x=diffs, y=y_pos, 
        xerr=[err_low, err_high], 
        fmt='s', color='#708090', markersize=7, capsize=0, linewidth=2
    )
    
    ax_plot.get_yaxis().set_visible(False)
    
    max_val = max([abs(r['ci_l']) for r in results] + [abs(r['ci_u']) for r in results]) * 1.1
    if max_val == 0: max_val = 1 
    ax_plot.set_xlim(-max_val, max_val)
    
    ax_plot.spines['left'].set_visible(False)
    ax_plot.spines['top'].set_visible(False)
    ax_plot.spines['right'].set_visible(False)
    ax_plot.spines['bottom'].set_linewidth(1.5)
    ax_plot.tick_params(axis='x', bottom=True, labelbottom=True, labelsize=10, width=1.5, length=5)
    ax_plot.set_xlabel("\u2190 Lower in Females  |  Higher in Females \u2192", fontsize=11, labelpad=15, fontweight='bold')
    ax_plot.grid(True, axis='x', linestyle='--', alpha=0.5)

    # RIGHT PANEL
    ax_right.set_xlim(0, 1)
    ax_right.axis('off')
    
    ax_right.text(0.5, n_items - 0.2, "P-Value", fontweight='bold', va='bottom', ha='center', fontsize=11)
    
    for i, res in enumerate(results):
        y = y_pos[i]
        p_str = f"{res['p']:.4g}" if res['p'] >= 0.0001 else "< 0.0001"
        ax_right.text(0.5, y, p_str, va='center', ha='center', fontsize=11)

    plt.subplots_adjust(wspace=0.1) 
    
    save_path = f"{save_dir}/graph2_median_differences.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"-> Successfully saved Graph 2 to: {save_path}")


# ==========================================================
# MAIN EXECUTION
# ==========================================================
def main():
    OUTPUT_DIR = '/Users/edwardyao/Documents/PURM/gender_project_output'
    
    df = load_and_filter_data()
    
    # --------------------------------------------------------
    # GRAPH 1: 11 CATEGORIES (ABSOLUTE MEDIANS)
    # ADDED TEMPORAL AND FRONTAL
    # --------------------------------------------------------
    median_categories = [
        ("Everyone with epilepsy", df['spike_rate'].values),
        ("Men with epilepsy", df[df['nlp_gender'] == 'M']['spike_rate'].values),
        ("Women with epilepsy", df[df['nlp_gender'] == 'F']['spike_rate'].values),
        ("Men w/ General", df[(df['nlp_gender'] == 'M') & (df['canonical_subtype'] == 'Generalized')]['spike_rate'].values),
        ("Women w/ General", df[(df['nlp_gender'] == 'F') & (df['canonical_subtype'] == 'Generalized')]['spike_rate'].values),
        ("Men w/ Focal", df[(df['nlp_gender'] == 'M') & (df['canonical_subtype'] == 'Focal')]['spike_rate'].values),
        ("Women w/ Focal", df[(df['nlp_gender'] == 'F') & (df['canonical_subtype'] == 'Focal')]['spike_rate'].values),
        ("Men w/ Temporal", df[(df['nlp_gender'] == 'M') & (df['canonical_subtype'] == 'Temporal')]['spike_rate'].values),
        ("Women w/ Temporal", df[(df['nlp_gender'] == 'F') & (df['canonical_subtype'] == 'Temporal')]['spike_rate'].values),
        ("Men w/ Frontal", df[(df['nlp_gender'] == 'M') & (df['canonical_subtype'] == 'Frontal')]['spike_rate'].values),
        ("Women w/ Frontal", df[(df['nlp_gender'] == 'F') & (df['canonical_subtype'] == 'Frontal')]['spike_rate'].values)
    ]
    
    median_results = []
    print("\nRunning memory-safe fast bootstrap for Medians (Graph 1)...")
    for label, data in median_categories:
        n_size = len(data) # Calculate Sample Size
        if n_size < 1:
            print(f"Skipping {label}: Insufficient data")
            continue
        med, cl, cu = bootstrap_median(data)
        # Passed 'n' into the results dict here
        median_results.append({'Group': label, 'n': n_size, 'median': med, 'ci_l': cl, 'ci_u': cu})

    plot_absolute_medians(median_results, save_dir=OUTPUT_DIR)
    
    # --------------------------------------------------------
    # GRAPH 2: 5 CATEGORIES (MEDIAN DIFFERENCES & P-VALUES)
    # ADDED TEMPORAL AND FRONTAL
    # --------------------------------------------------------
    diff_comparisons = [
        ("Overall", df),
        ("General", df[df['canonical_subtype'] == 'Generalized']),
        ("Focal", df[df['canonical_subtype'] == 'Focal']),
        ("Temporal", df[df['canonical_subtype'] == 'Temporal']),
        ("Frontal", df[df['canonical_subtype'] == 'Frontal'])
    ]
    
    diff_results = []
    print("\nRunning memory-safe fast bootstrap for Differences (Graph 2)...")
    for label, sub_df in diff_comparisons:
        f_group = sub_df[sub_df['nlp_gender'] == 'F']['spike_rate'].values
        m_group = sub_df[sub_df['nlp_gender'] == 'M']['spike_rate'].values
        
        n_f = len(f_group) # Female Sample Size
        n_m = len(m_group) # Male Sample Size
        
        if n_f < 3 or n_m < 3:
            print(f"Skipping {label}: Insufficient data")
            continue 
            
        diff, cl, cu, p = bootstrap_median_diff(f_group, m_group)
        # Passed both 'n_f' and 'n_m' into the results dict here
        diff_results.append({'Group': label, 'n_f': n_f, 'n_m': n_m, 'diff': diff, 'ci_l': cl, 'ci_u': cu, 'p': p})

    plot_median_differences(diff_results, save_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()