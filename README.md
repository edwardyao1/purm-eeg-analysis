This contains the code for running all analyses associated with the project evaluating sex differences in epilepsy spike burden and seizure frequency.

Requirements:
`spike_counts.csv` and `clinical_data_deidentified.csv`

`spike_counts.csv` contains a list of spike counts at varying SpikeNet probability thresholds for each EEG.

`clinical_data_deidentified.csv` contains clinical information for each patient. De-identified birth dates and dates of service are provided by date-shifting each date relative to the date of the patient's first clinic visit, defined to be Jan 1 2000. E.g., if a patient's date of birth is 1/1/1980 and their first visit is 1/1/2010, then their deidentified birth date is 1/1/1970. Each row is one EEG, and so the same patient may appear in multiple rows.

Python (3.x) and the necessary data science libraries (`pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`).

This codebase, located at: https://github.com/edwardyao1/purm-eeg-analysis

Create an output directory and a data directory in the paths noted in the scripts, and place both csv datasets in the data directory.

To run the analysis, navigate to the directory containing these scripts and run them individually. These functions will reproduce all figures, tables, and results text and save them in the output directory:

* **`mann_whitney_tests`**: Runs the Mann-Whitney U tests and generates the associated figures. Contains Figure 1A and Figure S2.
* **`multivariable_logistic_regression`**: Runs the main figures and analyses evaluating sex, epilepsy type, and age. Contains Table 1, Figure 1B, Table S2, Table S4, and Figure S2.
* **`multivariable_logistic_regression_menopause`**: Runs the supplemental analyses with the age split defined at 51 years old. Contains Figure S3.
* **`multivariable_logistic_regression_menopause_interaction`**: Runs the supplemental analyses with interaction terms utilizing the age split of 51 years old. Contains Table S3.
* * **`likelihood_ratio_testing`**: Runs the testing between the interaction terms of the main regression model. Contains Table S1.
* **`flowchart`**: Generates the supplemental figure of the patient cohort attrition and selection. Contains Figure S1.
