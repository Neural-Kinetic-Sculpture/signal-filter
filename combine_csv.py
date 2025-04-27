import pandas as pd

# File paths
eeg_file = 'D1_EEG.csv'
eog_file = 'D1_EOG.csv'
output_file = 'D1_EEG_EOG.csv'

# Read the CSVs
eeg_df = pd.read_csv(eeg_file, header=None)
eog_df = pd.read_csv(eog_file, header=None)

# Stack vertically (one under the other)
combined_df = pd.concat([eeg_df, eog_df], axis=0)

# Save to new CSV
combined_df.to_csv(output_file, index=False, header=False)

print(f"Combined file saved as {output_file}")
