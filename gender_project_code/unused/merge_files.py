import pandas as pd

# 1. Load the datasets
spike_df = pd.read_csv('/Users/edwardyao/Documents/PURM/data/spike_counts.csv')
clin_df = pd.read_csv('/Users/edwardyao/Documents/PURM/data/clinical_data_deidentified.csv')

# Optional: Drop identical duplicate rows in the clinical data 
clin_df_unique = clin_df.drop_duplicates()

# 2. Merge the two dataframes
# We use an 'inner' merge to keep only the patients that exist in BOTH files.
# left_on specifies the ID column in the first dataframe (spike_df)
# right_on specifies the ID column in the second dataframe (clin_df)
merged_df = pd.merge(
    spike_df, 
    clin_df_unique, 
    left_on='Patient', 
    right_on='patient_id', 
    how='inner'
)

# Keep only one demographic row per patient
clin_patients = clin_df.drop_duplicates(subset=['patient_id'])

# Merge with the spike data
merged_df = pd.merge(spike_df, clin_patients, left_on='Patient', right_on='patient_id', how='inner')

# 3. Save the merged dataset to a new CSV file
merged_df.to_csv('merged_spike_clinical_data.csv', index=False)

print("Data successfully merged and saved to 'merged_spike_clinical_data.csv'")