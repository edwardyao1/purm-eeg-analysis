import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ==========================================================
# LOAD DATA
# ==========================================================

# reading clinical data - contains the seizure frequencies and demographics
clin_df = pd.read_csv(
    "/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv"
)

# reading spike data - contains the spike counts and durations
spike_df = pd.read_csv(
    "/Users/edwardyao/Documents/PURM/data/spike_counts.csv"
)

# ==========================================================
# SAFE PARSE SEIZURE FREQUENCIES
# ==========================================================

def parse_seizure_frequencies(x):

    # check to make sure it's not NaN first
    if pd.isna(x):
        return []

    # safe parse with multiple layers of error handling to ensure we get a clean list of floats or an empty list
    try:
        vals = json.loads(x)
        out = []

        # iterate through the parsed values and convert to floats, ignoring any that can't be converted
        for v in vals:
            try:
                if v is None:
                    continue
                
                v = float(v)
                
                if np.isfinite(v):
                    out.append(v)
            except:
                continue

        return out

    except:
        return []

# ==========================================================
# CLINICAL FILTERS - same from MATLAB
# ==========================================================

# Filter for valid visit types
allowable_visits = {
    "CONSULT VISIT",
    "ESTABLISHED PATIENT VISIT",
    "FOLLOW-UP PATIENT CLINIC",
    "NEW PATIENT CLINIC",
    "NEW PATIENT VISIT",
    "NPV NEUROLOGY",
    "RETURN PATIENT VISIT"
}

# Apply visit type filter
clin_df = clin_df[
    clin_df["visit_type"].astype(str).apply(
        lambda v: any(a in v for a in allowable_visits)
    )
].copy()

# Filter for valid epilepsy types
bad_types = {
    "Non-Epileptic Seizure Disorder",
    "Uncertain if Epilepsy",
    "Unknown or MRN not found",
    ""
}

# Apply epilepsy type filter
clin_df = clin_df[
    ~clin_df["epilepsy_type"].isin(bad_types)
].copy()

# outpatient filter
acq = clin_df["acquired_on"].fillna("").astype(str).str.lower()
patient_class = clin_df["report_PATIENT_CLASS"].fillna("").astype(str).str.lower()
jay = clin_df["jay_in_or_out"].fillna("").astype(str).str.lower()

# Keep only outpatient EEGs or those acquired at SPE/Radnor
clin_df = clin_df[
    acq.str.contains("spe", na=False)
    | acq.str.contains("radnor", na=False)
    | (patient_class == "outpatient")
    | (jay == "out")
].copy()

# ==========================================================
# SPIKE RATE (spikes/hour) - SAFE
# ==========================================================

spike_df["Duration_sec"] = pd.to_numeric(spike_df["Duration_sec"], errors="coerce")
spike_df["count_0_46"] = pd.to_numeric(spike_df["count_0_46"], errors="coerce")

# Filter out sessions with non-positive duration or missing spike counts
spike_df = spike_df[
    (spike_df["Duration_sec"] > 0)
    & (spike_df["count_0_46"].notna())
].copy()

# Calculate spike rate safely, adding a small constant to avoid division by zero
spike_df["spike_rate"] = (
    spike_df["count_0_46"] * 3600 / spike_df["Duration_sec"]
)

spike_patient = (
    spike_df
    .groupby("Patient", as_index=False)
    .agg(mean_spike_rate=("spike_rate", "median"))
    .rename(columns={"Patient": "patient_id"})
)

# ==========================================================
# SEIZURE FREQUENCY PER PATIENT
# ==========================================================

# Apply clinical filters
clin_df["valid_sz_freqs"] = clin_df["sz_freqs"].apply(
    parse_seizure_frequencies
)

records = []

# Iterate through each patient and aggregate their seizure frequencies, taking the median if there are multiple entries
for pid, g in clin_df.groupby("patient_id"):

    vals = []
    for row in g["valid_sz_freqs"]:
        vals.extend(row)

    vals = [v for v in vals if np.isfinite(v)]

    if len(vals) == 0:
        continue

    records.append({
        "patient_id": pid,
        "mean_sz_freq": np.median(vals),
        "nlp_gender": g["nlp_gender"].iloc[0],
        "epilepsy_type": g["epilepsy_type"].iloc[0]
    })

seizure_df = pd.DataFrame(records)

seizure_df = seizure_df[
    seizure_df["nlp_gender"].isin(["M", "F"])
]

# ==========================================================
# MERGE
# ==========================================================

# Merge the spike rate and seizure frequency data on patient_id, keeping only patients that have both
df = seizure_df.merge(
    spike_patient,
    on="patient_id",
    how="inner"
).drop_duplicates("patient_id")

# ==========================================================
# REMOVE NON-FINITE BEFORE LOG
# ==========================================================

# Remove any patients with non-finite spike rates or seizure frequencies before log transformation
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["mean_spike_rate", "mean_sz_freq"])

df = df[
    (df["mean_spike_rate"] >= 0)
    & (df["mean_sz_freq"] >= 0)
].copy()

# ==========================================================
# LOG TRANSFORM (SAFE)
# ==========================================================

# Add a small constant before log transformation to avoid issues with zero values
df["log_spike"] = np.log10(df["mean_spike_rate"] + 0.01)
df["log_sz"] = np.log10(df["mean_sz_freq"] + 0.01)

# ==========================================================
# SPEARMAN FUNCTION
# ==========================================================

# Define a function to run the Spearman correlation and print results in a consistent format
def run(label, data):

    if len(data) < 5:
        print(f"{label:<25} N={len(data)}")
        return

    # compute Spearman correlation between log-transformed spike rates and seizure frequencies
    rho, p = spearmanr(
        data["mean_spike_rate"],
        data["mean_sz_freq"]
    )

    print(f"{label:<25} N={len(data):<5} rho={rho:.4f} p={p:.3g}")

# ==========================================================
# ANALYSES
# ==========================================================

print("\nSPIKE RATE VS SEIZURE FREQUENCY\n" + "-"*50)

run("Entire Cohort", df)
run("Male", df[df["nlp_gender"] == "M"])
run("Female", df[df["nlp_gender"] == "F"])

print("\nEpilepsy Types\n" + "-"*50)

# Run the analysis separately for each epilepsy type
for t in df["epilepsy_type"].dropna().unique():

    run(t, df[df["epilepsy_type"] == t])

# ==========================================================
# CLEAN DATA FOR PLOT
# ==========================================================

x = df["log_spike"].values
y = df["log_sz"].values

# remove any leftover NaNs just in case
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]

# ==========================================================
# LINEAR FIT (NUMPY ONLY)
# ==========================================================

m, b = np.polyfit(x, y, 1)

x_line = np.linspace(x.min(), x.max(), 200)
y_line = m * x_line + b

# ==========================================================
# SPEARMAN
# ==========================================================

rho, p = spearmanr(
    df["mean_spike_rate"],
    df["mean_sz_freq"]
)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(x, y, alpha=0.6)

plt.plot(
    x_line,
    y_line,
    color="red",
    linewidth=2,
    label=f"Fit slope = {m:.2f}"
)

plt.xlabel("Spike Rate (log(spikes/hour + 0.01))")
plt.ylabel("Seizure Frequency (log(seizures/hour + 0.01))")

plt.title(
    f"Spike Rate vs Seizure Frequency\n"
    f"Spearman ρ = {rho:.3f}, p = {p:.3g}"
)

plt.legend()
plt.tight_layout()
plt.show()