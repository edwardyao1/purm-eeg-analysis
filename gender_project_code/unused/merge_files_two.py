import pandas as pd

# Load the datasets
clinical_df = pd.read_csv("/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv")
spike_df = pd.read_csv("/Users/edwardyao/Documents/PURM/data/spike_counts.csv")

# Rename the columns in the clinical dataset so they match exactly
clinical_df = clinical_df.rename(columns={
    'patient_id': 'Patient', 
    'session_number': 'Session'
})

# Merge the datasets using 'on' since the column names now match exactly.
# This prevents pandas from creating duplicate columns.
merged_df = pd.merge(
    spike_df, 
    clinical_df, 
    on=['Patient', 'Session'], 
    how='outer'
)

# Save the merged dataframe to a new CSV file
merged_df.to_csv("/Users/edwardyao/Documents/PURM/data/merged_data_two.csv", index=False)

print(f"Merge complete.")
print(f"Rows in spike_counts: {len(spike_df)}")
print(f"Rows in clinical_data: {len(clinical_df)}")
print(f"Rows in merged_data: {len(merged_df)}")