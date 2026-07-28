import ast
import pandas as pd
import numpy as np

# Load data
clinical = pd.read_csv("/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv")
spikes = pd.read_csv("/Users/edwardyao/Documents/PURM/data/spike_counts.csv")

# Merge
df = spikes.merge(
    clinical,
    left_on=["Patient", "Session"],
    right_on=["patient_id", "session_number"],
    how="left"
)

print("\n" + "="*70)
print("Columns in the merged DataFrame:")
print(df.columns.tolist())
print("="*70)


print("\n" + "="*70)
print("Unique values in 'epilepsy_specific':")
print(sorted(df["epilepsy_specific"].dropna().unique()))
print("="*70)

#
# PRINTING UNIQUE VALUES IN THE COLUMNS
#

epilepsy_type_unique_values = df['epilepsy_type'].unique()
epilepsy_specific_unique_values = df['epilepsy_specific'].unique()
epilepsy_gender_unique_values = df['nlp_gender'].unique()
epilepsy_sex_unique_values = df['nlp_sex_identity'].unique()

print("Unique epilepsy types:", epilepsy_type_unique_values)
print("Unique epilepsy specifics:", epilepsy_specific_unique_values)
print("Unique epilepsy genders:", epilepsy_gender_unique_values)
print("Unique epilepsy sexes:", epilepsy_sex_unique_values)

# ==========================================================
# 1. RAW SEX COUNTS
# ==========================================================

known_sex = df[df["nlp_gender"].isin(["M", "F"])]

print("\n" + "="*70)
print("RAW SEX COUNTS (KNOWN SEX ONLY)")
print("="*70)
print(known_sex["nlp_gender"].value_counts())

# ==========================================================
# 2. APPLY INCLUSION CRITERIA
# ==========================================================

allowed_specific = [
    "Absence",
    "Juvenile Myoclonic",
    "Other Specified Generalized",
    "Unspecified Generalized",
    "Temporal Lobe",
    "Frontal Lobe",
    "Parietal/Occipital Lobe",
    "Multifocal",
    "Other Specified Focal",
    "Unlocalized Focal",
    "Combined Generalized and Focal",
    "GTCA",
    "Unclassified or Unspecified",
    "Unknown or MRN not found"
]

included = known_sex[
    known_sex["epilepsy_specific"].isin(allowed_specific)
].copy()

print(included["sz_freqs"].head())
print(included["sz_freqs"].dtype)

# ==========================================================
# 3. SEX COUNTS AFTER EXCLUSIONS
# ==========================================================

print("\n" + "="*70)
print("SEX COUNTS AFTER EXCLUSION CRITERIA")
print("="*70)
print(included["nlp_gender"].value_counts())

# ==========================================================
# 4. TOTALS FOR EACH CLASSIFICATION
# ==========================================================

criteria_totals = (
    included["epilepsy_specific"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\n" + "="*70)
print("TOTAL COUNT PER EPILEPSY CLASSIFICATION")
print("="*70)
print(criteria_totals)

# ==========================================================
# 5. MEN/WOMEN PER CLASSIFICATION
# ==========================================================

sex_by_class = pd.crosstab(
    included["epilepsy_specific"],
    included["nlp_gender"]
)

sex_by_class["Total"] = sex_by_class.sum(axis=1)

sex_by_class = sex_by_class.sort_values(
    "Total",
    ascending=False
)

print("\n" + "="*70)
print("MEN AND WOMEN PER EPILEPSY CLASSIFICATION")
print("="*70)
print(sex_by_class)

# ==========================================================
# OPTIONAL: PRETTY SUMMARY TABLE
# ==========================================================

summary = pd.crosstab(
    included["epilepsy_specific"],
    included["nlp_gender"]
).reset_index()

summary["Total"] = summary["M"] + summary["F"]

summary = summary.sort_values(
    "Total",
    ascending=False
)

print("\n" + "="*70)
print("FINAL SUMMARY TABLE")
print("="*70)
print(summary.to_string(index=False))

# ==========================================================
# 6. HIERARCHICAL SUMMARY TABLE
# ==========================================================

generalized = [
    "Absence",
    "Juvenile Myoclonic",
    "Other Specified Generalized",
    "Unspecified Generalized",
]

focal = [
    "Temporal Lobe",
    "Frontal Lobe",
    "Parietal/Occipital Lobe",
    "Multifocal",
    "Other Specified Focal",
    "Unlocalized Focal",
]

combined = [
    "Combined Generalized and Focal",
    "GTCA",
]

other = [
    "Unclassified or Unspecified",
    "Unknown or MRN not found",
]

rows = []


def add_group(group_name, subtype_list):

    rows.append([f"*** {group_name} ***", "", "", ""])

    total_f = 0
    total_m = 0

    for subtype in subtype_list:

        sub = included[
            included["epilepsy_specific"] == subtype
        ]

        f = (sub["nlp_gender"] == "F").sum()
        m = (sub["nlp_gender"] == "M").sum()

        total_f += f
        total_m += m

        rows.append([
            subtype,
            f,
            m,
            f + m
        ])

    rows.append([
        f"{group_name} TOTAL",
        total_f,
        total_m,
        total_f + total_m
    ])

    rows.append(["", "", "", ""])


add_group("GENERALIZED", generalized)
add_group("FOCAL", focal)
add_group("COMBINED GENERALIZED & FOCAL", combined)
add_group("OTHER", other)

hierarchical_table = pd.DataFrame(
    rows,
    columns=[
        "Classification",
        "Female",
        "Male",
        "Total"
    ]
)

print("\n" + "=" * 80)
print("HIERARCHICAL EPILEPSY CLASSIFICATION TABLE")
print("=" * 80)
print(hierarchical_table.to_string(index=False))

# ==========================================================
# 7. GROUP TOTALS ONLY
# ==========================================================

print("\n" + "=" * 80)
print("GROUP TOTALS")
print("=" * 80)

for group_name, subtype_list in {
    "GENERALIZED": generalized,
    "FOCAL": focal,
    "COMBINED GENERALIZED & FOCAL": combined,
    "OTHER": other,
}.items():

    sub = included[
        included["epilepsy_specific"].isin(subtype_list)
    ]

    f = (sub["nlp_gender"] == "F").sum()
    m = (sub["nlp_gender"] == "M").sum()

    print(
        f"{group_name:<30}"
        f" Female={f:<6}"
        f" Male={m:<6}"
        f" Total={f+m}"
    )

# ==========================================================
# 8. OVERALL STUDY SAMPLE
# ==========================================================

overall_f = (included["nlp_gender"] == "F").sum()
overall_m = (included["nlp_gender"] == "M").sum()

print("\n" + "=" * 80)
print("OVERALL INCLUDED STUDY SAMPLE")
print("=" * 80)
print(f"Female: {overall_f}")
print(f"Male:   {overall_m}")
print(f"Total:  {overall_f + overall_m}")

# ==========================================================
# 9. SEIZURE FREQUENCY AVAILABILITY
# ==========================================================

def has_seizure_frequency(x):

    if pd.isna(x):
        return False

    try:
        vals = ast.literal_eval(
            str(x).replace("null", "None")
        )

        if not isinstance(vals, list):
            return False

        return any(v is not None for v in vals)

    except Exception:
        return False


df["has_sz_freq"] = df["sz_freqs"].apply(
    has_seizure_frequency
)

included["has_sz_freq"] = included["sz_freqs"].apply(
    has_seizure_frequency
)

# ==========================================================
# 10. AGE CALCULATION
# ==========================================================

# ==========================================================
# AGE CALCULATION
# Assume all years are in the 1900s
# Example: 12/9/44 = December 9, 1944
# ==========================================================

# ----------------------------
# 1. Parse birth dates safely
# ----------------------------
included["deid_birth_date_parsed"] = pd.to_datetime(
    included["deid_birth_date"],
    format="%m/%d/%y",
    errors="coerce"
)

# Fix century assumption: force 00–99 into 1900s
mask_2000s = included["deid_birth_date_parsed"].dt.year >= 2000
included.loc[mask_2000s, "deid_birth_date_parsed"] -= pd.DateOffset(years=100)

# ----------------------------
# 2. Parse visit timestamps
# ----------------------------
included["start_time_parsed"] = pd.to_datetime(
    included["start_time_deid"],
    errors="coerce"
)

# ----------------------------
# 3. Compute age (years)
# ----------------------------
included["age"] = (
    (included["start_time_parsed"] - included["deid_birth_date_parsed"])
    .dt.total_seconds() / (365.25 * 24 * 60 * 60)
)

# ==========================================================
# 11. COHORT FLOW
# ==========================================================

print("\n" + "="*80)
print("COHORT FLOW")
print("="*80)

sex_cohort = df[
    df["nlp_gender"].isin(["M", "F"])
]

freq_cohort = sex_cohort[
    sex_cohort["has_sz_freq"]
]

freq_type_cohort = included[
    included["has_sz_freq"]
]

for name, cohort in [
    ("Known sex + EEG", sex_cohort),
    ("Known sex + EEG + seizure frequency", freq_cohort),
    ("Known sex + EEG + seizure frequency + epilepsy type",
     freq_type_cohort),
]:

    female = (cohort["nlp_gender"] == "F").sum()
    male = (cohort["nlp_gender"] == "M").sum()

    print(f"\n{name}")
    print(f"Female: {female}")
    print(f"Male:   {male}")
    print(f"Total:  {female + male}")
    
# ==========================================================
# 12. MAJOR EPILEPSY GROUPS
# ==========================================================

print("\n" + "="*80)
print("GENERALIZED VS FOCAL VS COMBINED")
print("="*80)

for group_name, subtype_list in {
    "Generalized": generalized,
    "Focal": focal,
    "Combined": combined,
    "Other": other,
}.items():

    temp = included[
        included["epilepsy_specific"].isin(subtype_list)
    ]

    female = (temp["nlp_gender"] == "F").sum()
    male = (temp["nlp_gender"] == "M").sum()

    print(
        f"{group_name:<15}"
        f" Female={female:<6}"
        f" Male={male:<6}"
        f" Total={female + male}"
    )

print("\nNOTE:")
print(
    "Combined generalized and focal cases are typically "
    "excluded from direct generalized-vs-focal analyses."
)

# ==========================================================
# 13. AGE SUMMARY
# ==========================================================

# age summary checks 

print(
    included[
        [
            "deid_birth_date",
            "start_time_deid",
            "age"
        ]
    ].head(20)
)
print(included["age"].describe())

print(
    included[
        (included["age"] < 0) |
        (included["age"] > 120)
    ][
        [
            "Patient",
            "deid_birth_date",
            "start_time_deid",
            "age"
        ]
    ]
)

print("\n" + "="*80)
print("AGE SUMMARY")
print("="*80)

print(
    included["age"]
    .describe()
    [["count","mean","std","min","50%","max"]]
)

print("\nAGE BY SEX")

print(
    included
    .groupby("nlp_gender")["age"]
    .describe()
    [["count","mean","std","min","50%","max"]]
)

# ==========================================================
# 14. UNIQUE PATIENT COHORT
# ==========================================================

patients = included.sort_values(
    "start_time_deid"
).drop_duplicates(
    "Patient",
    keep="first"
)

print("\n" + "="*80)
print("UNIQUE PATIENTS")
print("="*80)

print(patients["nlp_gender"].value_counts())

print("\nAGE BY SEX (PATIENT LEVEL)")

print(
    patients
    .groupby("nlp_gender")["age"]
    .describe()
    [["count","mean","std","min","50%","max"]]
)

print("\nEPILEPSY GROUPS (PATIENT LEVEL)")

for group_name, subtype_list in {
    "Generalized": generalized,
    "Focal": focal,
    "Combined": combined,
    "Other": other,
}.items():

    temp = patients[
        patients["epilepsy_specific"].isin(subtype_list)
    ]

    print(f"{group_name}: {len(temp)}")
    
# ==========================================================
# 15. SPIKE RATE SAMPLE SIZES BY SEX
# ==========================================================

# NOTE: Update 'spike_rate' to match the actual column name from spike_counts.csv
spike_col = 'count_0_46'  # Example column name for spike rate

print("\n" + "="*80)
print("SPIKE RATE COHORT SUMMARY")
print("="*80)

if spike_col in included.columns:
    
    # Filter for valid, non-zero spike rates
    # (Adjust the condition if your data includes 0 as a valid spike rate you want to count)
    spike_cohort = included[
        (included[spike_col].notna()) & 
        (included[spike_col] > 0)
    ]
    
    # Session-level counts
    spike_f_session = (spike_cohort["nlp_gender"] == "F").sum()
    spike_m_session = (spike_cohort["nlp_gender"] == "M").sum()
    
    print("SESSION LEVEL (Total EEGs/Visits with spikes):")
    print(f"Female: {spike_f_session}")
    print(f"Male:   {spike_m_session}")
    print(f"Total:  {spike_f_session + spike_m_session}")
    
    # Patient-level counts
    unique_patients_spikes = spike_cohort.drop_duplicates(
        subset="Patient", 
        keep="first"
    )
    
    spike_f_patient = (unique_patients_spikes["nlp_gender"] == "F").sum()
    spike_m_patient = (unique_patients_spikes["nlp_gender"] == "M").sum()
    
    print("\nPATIENT LEVEL (Unique individuals with spikes):")
    print(f"Female: {spike_f_patient}")
    print(f"Male:   {spike_m_patient}")
    print(f"Total:  {spike_f_patient + spike_m_patient}")

else:
    print(f"ERROR: Column '{spike_col}' not found in the merged DataFrame.")
    print("Please update the 'spike_col' variable with the correct column name.")
    
# ==========================================================
# 15. SPIKE RATE SAMPLE SIZES BY SEX
# ==========================================================
spike_col = 'count_0_46'
print("\n" + "="*80)
print("SPIKE RATE COHORT SUMMARY")
print("="*80)
if spike_col in included.columns:
    
    spike_cohort = included[
        (included[spike_col].notna()) & 
        (included[spike_col] > 0)
    ]

    # --- SESSION LEVEL ---
    spike_f_session = (spike_cohort["nlp_gender"] == "F").sum()
    spike_m_session = (spike_cohort["nlp_gender"] == "M").sum()
    
    print("SESSION LEVEL (Total EEGs/Visits with spikes):")
    print(f"Female: {spike_f_session}")
    print(f"Male:   {spike_m_session}")
    print(f"Total:  {spike_f_session + spike_m_session}")

    # --- PATIENT LEVEL ---
    unique_patients_spikes = spike_cohort.drop_duplicates(subset="Patient", keep="first")
    spike_f_patient = (unique_patients_spikes["nlp_gender"] == "F").sum()
    spike_m_patient = (unique_patients_spikes["nlp_gender"] == "M").sum()
    
    print("\nPATIENT LEVEL (Unique individuals with spikes):")
    print(f"Female: {spike_f_patient}")
    print(f"Male:   {spike_m_patient}")
    print(f"Total:  {spike_f_patient + spike_m_patient}")

    # --- BY EPILEPSY TYPE (PATIENT LEVEL) ---
    epilepsy_type_map = {
        "Generalized":  ["General"],
        "Focal":        ["Focal"],
        "Combined":     ["Combined Generalized and Focal"],
        "Other":        [
            "Unclassified or Unspecified",
            "Uncertain if Epilepsy",
            "Non-Epileptic Seizure Disorder",
            "Unknown or MRN not found"
        ]
    }

    print("\nPATIENT LEVEL BY EPILEPSY TYPE:")
    print(f"{'Type':<12} {'Female':>8} {'Male':>8} {'Total':>8}")
    print("-" * 40)

    for group_label, type_values in epilepsy_type_map.items():
        subset = unique_patients_spikes[
            unique_patients_spikes["epilepsy_type"].isin(type_values)
        ]
        f = (subset["nlp_gender"] == "F").sum()
        m = (subset["nlp_gender"] == "M").sum()
        print(f"{group_label:<12} {f:>8} {m:>8} {f+m:>8}")

    # Totals row
    print("-" * 40)
    print(f"{'TOTAL':<12} {spike_f_patient:>8} {spike_m_patient:>8} {spike_f_patient+spike_m_patient:>8}")

else:
    print(f"ERROR: Column '{spike_col}' not found in the merged DataFrame.")
    print("Please update the 'spike_col' variable with the correct column name.")