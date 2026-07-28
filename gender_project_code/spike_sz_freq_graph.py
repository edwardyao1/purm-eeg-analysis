import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore') # Suppress nan-mean warnings for empty slices

# ==========================================================
# LOAD DATA
# ==========================================================
clinical_df = pd.read_csv("/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv")
spike_df = pd.read_csv("/Users/edwardyao/Documents/PURM/data/spike_counts.csv")

# Standardize key names for easier merging
clinical_df = clinical_df.rename(columns={"patient_id": "Patient", "session_number": "Session"})

# ==========================================================
# BOOTSTRAP SPEARMAN CONFIDENCE INTERVALS
# ==========================================================
def bootstrap_spearman_ci(x, y, n_boot=5000, alpha=0.05):
    # Calculates 95% CI for Spearman rho using bootstrapping.
    x, y = np.array(x), np.array(y)
    n = len(x)
    if n < 3: return np.nan, np.nan, np.nan
    
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        boot_rhos[i] = r
        
    ci_lo = np.nanpercentile(boot_rhos, 100 * (alpha / 2))
    ci_hi = np.nanpercentile(boot_rhos, 100 * (1 - alpha / 2))
    rho_hat, _ = spearmanr(x, y)
    
    return rho_hat, ci_lo, ci_hi

# ==========================================================
# 1. OUTPATIENT & ROUTINE FILTER (STRICT INNER JOIN)
# ==========================================================
# Filter clinical data for Outpatient definitions
acq = clinical_df["acquired_on"].fillna("").astype(str).str.lower()
p_class = clinical_df["report_PATIENT_CLASS"].fillna("").astype(str).str.lower()
jay = clinical_df["jay_in_or_out"].fillna("").astype(str).str.lower()

is_outpt = (
    acq.str.contains("spe") | 
    acq.str.contains("radnor") |
    (p_class == "outpatient") | 
    (jay == "out")
)

clinical_output = clinical_df[is_outpt][['Patient', 'Session']]

# Filter spike data for Routine definitions (positive duration AND <= 4 hours)
spike_df["Duration_sec"] = pd.to_numeric(spike_df["Duration_sec"], errors="coerce")
spike_df["count_0_46"] = pd.to_numeric(spike_df["count_0_46"], errors="coerce")

# Define routine as >0 and <=14400 seconds, and must have a valid count_0_46 value (not NaN)
is_routine = (spike_df["Duration_sec"] > 0) & (spike_df["Duration_sec"] <= 14400) & (spike_df["count_0_46"].notna())
spike_routine = spike_df[is_routine][['Patient', 'Session']]

# Inner join on BOTH Patient and Session to get validated keys exactly like MATLAB
valid_sessions = clinical_output.merge(spike_routine, on=['Patient', 'Session'], how='inner').drop_duplicates()

# Apply the strict cohort filter to both core DataFrames
clinical_df = clinical_df.merge(valid_sessions, on=['Patient', 'Session'], how='inner')
spike_df = spike_df.merge(valid_sessions, on=['Patient', 'Session'], how='inner')

# ==========================================================
# 2. PATIENT-LEVEL SPIKE RATE
# ==========================================================
# Calculate spike rate per session and then average across sessions for each patient
spike_df["spike_rate"] = spike_df["count_0_46"] * 3600 / spike_df["Duration_sec"]
spike_patient = spike_df.groupby("Patient", as_index=False).agg(mean_spike_rate=("spike_rate", "mean"))

# ==========================================================
# 3. VISIT-LEVEL PARSING & ZERO-IMPUTATION (RULE 1)
# ==========================================================
# Visits that are allowed that was taken from MATLAB
allowable_visits = {
    "CONSULT VISIT", 
    "ESTABLISHED PATIENT VISIT", 
    "FOLLOW-UP PATIENT CLINIC", 
    "NEW PATIENT CLINIC", 
    "NEW PATIENT VISIT", 
    "NPV MANAGEMENT DURING COVID-19", 
    "NPV NEUROLOGY", 
    "RETURN ANNUAL VISIT", 
    "RETURN PATIENT EXTENDED", 
    "RETURN PATIENT VISIT", 
    "RPV MANAGEMENT DURING COVID-19", 
    "TELEHEALTH VIDEO VISIT RETURN"
}

def safe_json_load(s):
    # Robust JSON parser that tolerates MATLAB's NaN/null formatting and artifacts.
    if pd.isna(s): return []
    s = str(s).strip()
    if s in ["", "[]", "<missing>", "null"]: return []
    try:
        return json.loads(s)
    except Exception:
        try:
            return eval(s.replace('null', 'None').replace('NaN', 'None'))
        except Exception:
            return []

visit_records = []

# Iterate through clinical records and parse visit-level data according to MATLAB rules
for idx, row in clinical_df.iterrows():
    pid = row['Patient']
    
    vt_list = safe_json_load(row['visit_type'])
    date_list = safe_json_load(row['visit_dates_deid'])
    freq_list = safe_json_load(row['sz_freqs'])
    hassz_list = safe_json_load(row['visit_hasSz'])
    
    n = min(len(vt_list), len(date_list), len(freq_list), len(hassz_list))
    
    # Loop through visits and apply MATLAB rules for filtering
    for i in range(n):
        if vt_list[i] in allowable_visits:
            
            # 1. Safely parse Seizure Frequency
            try:
                f_val = float(freq_list[i]) if freq_list[i] is not None else np.nan
            except (ValueError, TypeError):
                f_val = np.nan
                
            # 2. Safely parse HasSz
            try:
                h_val = float(hassz_list[i]) if hassz_list[i] is not None else np.nan
            except (ValueError, TypeError):
                h_val = np.nan
            
            # 3. Apply MATLAB rules (negative freqs -> NaN, HasSz == 2 -> NaN)
            if f_val < 0 or not np.isfinite(f_val): 
                f_val = np.nan
            if h_val == 2 or not np.isfinite(h_val): 
                h_val = np.nan 
            
            visit_records.append({
                'Patient': pid, 
                'visit_date': date_list[i], 
                'freq': f_val, 
                'hassz': h_val
            })

visits_df = pd.DataFrame(visit_records)

# Aggregate to unique Patient + Date combos to resolve duplicates from multiple EEGs
def agg_visit(g):
    return pd.Series({'freq_agg': g['freq'].mean(skipna=True), 'has_agg': g['hassz'].max()})

vuniq = visits_df.groupby(['Patient', 'visit_date']).apply(agg_visit).reset_index()

# MATLAB Rule 1: HasSz==0 with no documented frequency -> impute SzFreq=0
mask_rule1 = vuniq['freq_agg'].isna() & (vuniq['has_agg'] == 0)
vuniq.loc[mask_rule1, 'freq_agg'] = 0

# Mean across all valid visits for each patient
pat_sz = vuniq.dropna(subset=['freq_agg']).groupby('Patient')['freq_agg'].mean().reset_index()
pat_sz = pat_sz.rename(columns={'freq_agg': 'mean_sz_freq'})

# ==========================================================
# 4. RECREATE MATLAB SUBTYPE LOGIC (EpiType3) & BAD TYPES
# ==========================================================
def first_valid(series):
    valid = series.dropna().replace("", np.nan).dropna()
    return valid.iloc[0] if len(valid) > 0 else ""

pat_info = clinical_df.groupby('Patient').agg({
    'nlp_gender': first_valid,
    'epilepsy_type': first_valid,
    'epilepsy_specific': first_valid
}).reset_index()

# Filter out Bad Types at the Patient Aggregation level
bad_types = {"non-epileptic seizure disorder", "uncertain if epilepsy", "unknown or mrn not found", ""}
pat_info = pat_info[~pat_info['epilepsy_type'].str.lower().str.strip().isin(bad_types)]

# Define canonical subtype based on epilepsy_specific and epilepsy_type fields
def categorize_epilepsy(row):
    spec = str(row.get('epilepsy_specific', '')).lower().strip()
    etype = str(row.get('epilepsy_type', '')).lower().strip()
    
    if 'temporal' in spec: return 'Temporal'
    elif 'frontal' in spec: return 'Frontal'
    elif etype == 'general': return 'General'
    else: return 'Other'

pat_info['EpiType3'] = pat_info.apply(categorize_epilepsy, axis=1)

# ==========================================================
# 5. MERGE & CLEAN COHORT
# ==========================================================
# Merge patient-level seizure frequency, spike rate, and clinical info into a single DataFrame for plotting
df = pat_sz.merge(pat_info, on="Patient", how="inner").merge(spike_patient, on="Patient", how="inner")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_spike_rate", "mean_sz_freq"])
df = df[(df["mean_spike_rate"] >= 0) & (df["mean_sz_freq"] >= 0)].copy()

# ==========================================================
# 6. PLOT - MATCHING MATLAB EXACTLY
# ==========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
axes = axes.flatten()

plot_configs = [
    {"label": "All epilepsy", "type": "All", "color": "#B0B0B0", "line_color": "black", "title_prefix": "A."},
    {"label": "Frontal", "type": "Frontal", "color": "#EBA715", "line_color": "#EBA715", "title_prefix": "B."},
    {"label": "Temporal", "type": "Temporal", "color": "#E46A42", "line_color": "#E46A42", "title_prefix": "C."},
    {"label": "General", "type": "General", "color": "#2082C5", "line_color": "#2082C5", "title_prefix": "D."}
]

# Calculate global epsilons dynamically based on min positive values
x_all = df["mean_sz_freq"].values
y_all = df["mean_spike_rate"].values

# To avoid log(0), we set eps to half the minimum positive value in each dataset, or a small constant if no positives exist.
minpos_sz = np.nanmin(x_all[x_all > 0]) if np.any(x_all > 0) else 1e-6
minpos_rate = np.nanmin(y_all[y_all > 0]) if np.any(y_all > 0) else 1e-6

# Set eps to half the minimum positive value to ensure zeros are plotted just below the smallest positive point on the log scale
eps_sz = 0.5 * minpos_sz
eps_rate = 0.5 * minpos_rate
log_eps_sz = np.log10(eps_sz)
log_eps_rate = np.log10(eps_rate)

# Loop through each subplot configuration and plot accordingly
for i, config in enumerate(plot_configs):
    ax = axes[i]

    subset = df if config["type"] == "All" else df[df["EpiType3"] == config["type"]]
    n_samples = len(subset)

    if n_samples < 3:
        ax.axis('off')
        continue

    x_raw = subset["mean_sz_freq"].values
    y_raw = subset["mean_spike_rate"].values

    rho, p = spearmanr(x_raw, y_raw)
    _, ci_lo, ci_hi = bootstrap_spearman_ci(x_raw, y_raw)

    # Offset zeros
    log_x = np.log10(np.where(x_raw <= 0, eps_sz, x_raw))
    log_y = np.log10(np.where(y_raw <= 0, eps_rate, y_raw))

    is_zx = x_raw == 0
    is_zy = y_raw == 0

    # 1. Plot normal strictly positive points
    ax.scatter(log_x[~is_zx & ~is_zy], log_y[~is_zx & ~is_zy], color=config["color"], alpha=0.35, s=15, edgecolors='none')
    
    # 2. Plot exact zeros with asterisks directly on the axes lines
    if np.any(is_zx & ~is_zy): ax.plot(log_x[is_zx & ~is_zy], log_y[is_zx & ~is_zy], '*', color=config["color"], markersize=6)
    if np.any(~is_zx & is_zy): ax.plot(log_x[~is_zx & is_zy], log_y[~is_zx & is_zy], '*', color=config["color"], markersize=6)
    if np.any(is_zx & is_zy):  ax.plot(log_x[is_zx & is_zy], log_y[is_zx & is_zy], '*', color=config["color"], markersize=8)

    # 3. Fit line ONLY on strictly positive values, bounded exactly to that subset's min/max bounds
    if np.sum(~is_zx & ~is_zy) >= 3:
        x_pos = log_x[~is_zx & ~is_zy]
        y_pos = log_y[~is_zx & ~is_zy]
        
        m, b = np.polyfit(x_pos, y_pos, 1)
        
        x_line = np.linspace(x_pos.min(), x_pos.max(), 200) 
        y_line = m * x_line + b
        ax.plot(x_line, y_line, color=config["line_color"], linewidth=1.5)

    # AXIS FORMATTING
    ax.axvline(log_eps_sz, color='gray', linestyle=':', linewidth=1)
    ax.axhline(log_eps_rate, color='gray', linestyle=':', linewidth=1)

    custom_x_ticks = [log_eps_sz, 0, 1, 2, 3]
    custom_y_ticks = [log_eps_rate, 0, 1, 2, 3]
    custom_labels = ["0", "1", "10", "100", "1000"]

    ax.set_xticks(custom_x_ticks)
    ax.set_xticklabels(custom_labels)
    ax.set_yticks(custom_y_ticks)
    ax.set_yticklabels(custom_labels)

    ax.set_xlim([log_eps_sz - 0.5, 3.8])
    ax.set_ylim([log_eps_rate - 0.5, 3.2])

    ax.set_xlabel("Seizures per month (log scale)", fontsize=13)
    ax.set_ylabel("Spikes per hour (log scale)", fontsize=13)
    ax.set_title(f"{config['title_prefix']} {config['label']} (N={n_samples})", fontsize=15, fontweight='bold')

    if config["type"] != "All":
        p_bonf = min(p * 3, 1.0)
        p_str = "<0.001" if p_bonf < 0.001 else f"={p_bonf:.2g}"
        stats_text = rf"$\rho$={rho:.2f} [{ci_lo:.2f}-{ci_hi:.2f}], $p_{{bonf}}${p_str}"
    else:
        p_str = "<0.001" if p < 0.001 else f"={p:.2g}"
        stats_text = rf"$\rho$={rho:.2f} [{ci_lo:.2f}-{ci_hi:.2f}], p{p_str}"
        
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', fontsize=12, fontweight='bold')

    ax.grid(True, linestyle='-', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.savefig("/Users/edwardyao/Documents/PURM/gender_project_output/spike_sz_freq_graph.png", dpi=300, bbox_inches="tight")

plt.tight_layout()
plt.show()