import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress minor seaborn warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ==========================================================
# UTILITIES & DATA PIPELINE
# ==========================================================
def canonical_subtype(row):
    etype = str(row.get("epilepsy_type", "")).lower()
    espec = str(row.get("epilepsy_specific", "")).lower()

    if "temporal" in espec: return "Temporal"
    if "frontal" in espec: return "Frontal"
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

    # Aggregate to Patient Level
    patient_df = df.groupby(["Patient", "nlp_gender", "canonical_subtype"], dropna=False).agg(
        total_spikes=("count_0_46", "sum"),
        total_duration=("Duration_sec", "sum")
    ).reset_index()

    patient_df = patient_df[patient_df['total_duration'] > 0].copy()
    patient_df = patient_df.dropna(subset=['canonical_subtype', 'nlp_gender']).copy()

    # Standardize nomenclature
    patient_df['nlp_gender'] = patient_df['nlp_gender'].replace({'M': 'Male', 'F': 'Female'})
    patient_df['canonical_subtype'] = patient_df['canonical_subtype'].replace({'General': 'Generalized'})

    # CREATE "ALL EPILEPSY" CATEGORY
    # We duplicate the strictly filtered cohort and label them all as "All Epilepsy"
    # so they appear as an aggregate group on the far left of the graph
    all_epi_df = patient_df.copy()
    all_epi_df['canonical_subtype'] = 'All Epilepsy'
    patient_df = pd.concat([all_epi_df, patient_df], ignore_index=True)

    # Calculate spike rates
    patient_df["spike_rate_per_hour"] = (patient_df["total_spikes"] / patient_df["total_duration"] * 3600)
    
    # Version 1: With Zeros (offset by +0.01 so exact 0 = -2.0)
    patient_df['log_rate_with_zeros'] = np.log10(patient_df['spike_rate_per_hour'] + 0.01)
    
    # Version 2: Without Zeros (Strictly > 0)
    patient_df['log_rate_no_zeros'] = patient_df['spike_rate_per_hour'].apply(lambda x: np.log10(x) if x > 0 else np.nan)

    return patient_df


# ==========================================================
# GRAPH GENERATION & DYNAMIC LABELING FUNCTIONS
# ==========================================================
X_BASE_ORDER = ['All Epilepsy', 'Generalized', 'Focal', 'Temporal', 'Frontal']

def prepare_dynamic_labels(df, target_col):
    """
    Dynamically calculates N for the specific plot being generated.
    Returns a dataframe with labeled columns, plus the exact rendering orders.
    """
    df_plot = df.dropna(subset=[target_col]).copy()
    
    # Generate X-axis labels with N
    x_mapped = []
    for st in X_BASE_ORDER:
        n_st = len(df_plot[df_plot['canonical_subtype'] == st])
        new_st = f"{st}\n(N={n_st})"
        df_plot.loc[df_plot['canonical_subtype'] == st, 'x_label'] = new_st
        x_mapped.append(new_st)
        
    # Generate Legend labels with N (using 'All Epilepsy' to avoid double counting)
    h_mapped = []
    palette_mapped = {}
    for g, color in [('Male', '#4C72B0'), ('Female', '#C44E52')]:
        n_g = len(df_plot[(df_plot['nlp_gender'] == g) & (df_plot['canonical_subtype'] == 'All Epilepsy')])
        new_g = f"{g} (N={n_g})"
        df_plot.loc[df_plot['nlp_gender'] == g, 'hue_label'] = new_g
        h_mapped.append(new_g)
        palette_mapped[new_g] = color
        
    return df_plot, x_mapped, h_mapped, palette_mapped


def format_graph(title, ylabel):
    """Standardizes axes, legends, and styling."""
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epilepsy Subtype', fontsize=13, labelpad=10)
    plt.ylabel(ylabel, fontsize=13, labelpad=10)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend(handles[:2], labels[:2], title='Sex (Total Cohort)', title_fontsize='12', fontsize='11', loc='upper right')
    
    plt.tight_layout()


def plot_line_graph(df, save_dir):
    """Creates a standalone Line Graph (Means & 95% CIs)."""
    df_plot, x_ord, h_ord, pal = prepare_dynamic_labels(df, 'log_rate_with_zeros')

    plt.figure(figsize=(12, 6), facecolor='white')
    sns.set_theme(style="whitegrid")
    
    sns.pointplot(
        data=df_plot, x='x_label', y='log_rate_with_zeros', hue='hue_label',
        order=x_ord, hue_order=h_ord, dodge=0.2,
        markers=['o', 's'], linestyles=['-', '--'],
        palette=pal, scale=1.2, errwidth=2, capsize=0.1
    )
    
    # Add vertical separator line between All Epilepsy and Specific Subtypes
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5, zorder=0)

    format_graph('Trend of Mean Spike Rates by Subtype', 'Spikes per hour (log10 scale, zeros included)')
    
    path = os.path.join(save_dir, "line_graph_means.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> Saved: {path}")


def plot_boxplot_with_zeros(df, save_dir):
    """Creates a Boxplot + Dots where Zeros are included."""
    df_plot, x_ord, h_ord, pal = prepare_dynamic_labels(df, 'log_rate_with_zeros')

    plt.figure(figsize=(14, 7), facecolor='white')
    sns.set_theme(style="whitegrid")
    
    sns.boxplot(
        data=df_plot, x='x_label', y='log_rate_with_zeros', hue='hue_label',
        order=x_ord, hue_order=h_ord, dodge=True,
        boxprops={'facecolor':'none', 'edgecolor': '#555555', 'linewidth': 1.5}, 
        whiskerprops={'color': '#555555', 'linewidth': 1.5},
        capprops={'color': '#555555', 'linewidth': 1.5},
        medianprops={'color': 'black', 'linewidth': 2},
        showfliers=False, whis=[5, 95]
    )
    
    sns.stripplot(
        data=df_plot, x='x_label', y='log_rate_with_zeros', hue='hue_label',
        order=x_ord, hue_order=h_ord, dodge=True, alpha=0.4, size=4, jitter=0.2,
        palette=pal
    )
    
    # Vertical separator
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5, zorder=0)
    
    # Zero baseline
    plt.axhline(-2.0, color='gray', linestyle=':', alpha=0.5, zorder=0)
    plt.text(4.4, -1.9, 'Zero Spikes\nBaseline', color='gray', fontsize=10, va='bottom', ha='right')
    
    format_graph('Distribution of Spike Rates (INCLUDING Zero-Spike Patients)', 'Spikes per hour (log10 scale)')
    
    path = os.path.join(save_dir, "boxplot_WITH_zeros.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> Saved: {path}")


def plot_boxplot_without_zeros(df, save_dir):
    """Creates a Boxplot + Dots where strictly zero spikes are DROPPED."""
    df_plot, x_ord, h_ord, pal = prepare_dynamic_labels(df, 'log_rate_no_zeros')

    plt.figure(figsize=(14, 7), facecolor='white')
    sns.set_theme(style="whitegrid")
    
    sns.boxplot(
        data=df_plot, x='x_label', y='log_rate_no_zeros', hue='hue_label',
        order=x_ord, hue_order=h_ord, dodge=True,
        boxprops={'facecolor':'none', 'edgecolor': '#555555', 'linewidth': 1.5}, 
        whiskerprops={'color': '#555555', 'linewidth': 1.5},
        capprops={'color': '#555555', 'linewidth': 1.5},
        medianprops={'color': 'black', 'linewidth': 2},
        showfliers=False, whis=[5, 95]
    )
    
    sns.stripplot(
        data=df_plot, x='x_label', y='log_rate_no_zeros', hue='hue_label',
        order=x_ord, hue_order=h_ord, dodge=True, alpha=0.4, size=4, jitter=0.2,
        palette=pal
    )
    
    # Vertical separator
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5, zorder=0)
    
    format_graph('Distribution of Spike Rates (EXCLUDING Zero-Spike Patients)', 'Spikes per hour (log10 scale)')
    
    path = os.path.join(save_dir, "boxplot_WITHOUT_zeros.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> Saved: {path}")


# ==========================================================
# MAIN EXECUTION
# ==========================================================
def main():
    OUTPUT_DIR = '/Users/edwardyao/Documents/PURM/gender_project_output'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print("Loading, filtering data, and building 'All Epilepsy' cohort...")
    df = load_and_filter_data()
    
    print("\nGenerating separated plots with dynamic Sample Sizes (N)...")
    plot_line_graph(df, save_dir=OUTPUT_DIR)
    plot_boxplot_with_zeros(df, save_dir=OUTPUT_DIR)
    plot_boxplot_without_zeros(df, save_dir=OUTPUT_DIR)
    
    print("\nAll plots generated and saved successfully!")

if __name__ == "__main__":
    main()