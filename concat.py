import pandas as pd

# Load the two CSV files
df1 = pd.read_csv('ballet_features2_1.csv')
df2 = pd.read_csv('ballet_features2_2.csv')

# Concatenate them 
combined_df = pd.concat([df1, df2], ignore_index=True)  # ignore_index resets row numbers

# Save to a new file 
combined_df.to_csv('ballet_features2_combined.csv', index=False)

print("Concatenation complete! New file: ballet_features2_combined.csv")