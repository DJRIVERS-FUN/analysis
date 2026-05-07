import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
FILE = 'data/381completeanalysis.xlsx'
df = pd.read_excel(FILE)

# Ensure output folders exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Variables of interest
# These names match the column names in 381completeanalysis.xlsx.
variables = {
    'GPARAW': 'GPA',
    'CMOR': 'Conscious Monitoring of Responsibilities',
    'VOBL': 'Verification of Fulfilment of Objectives',
    'MONITOR': 'MAI Monitoring',
    'EVALUAT': 'MAI Evaluation',
    'PLANNING': 'MAI Planning',
    'Distress': 'Psychological Distress'
}

# Select and rename variables for clearer output
data = df[list(variables.keys())].rename(columns=variables)

# Correlation matrix
corr = data.corr()

# Save matrix
corr.to_csv('outputs/correlation_matrix.csv')

# Plot heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson r'}
)

plt.title('Correlation Matrix: GPA, Executive Function, Metacognition, and Distress')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=300)

print('\nCorrelation matrix saved to outputs/correlation_matrix.csv')
print('Figure saved to figures/correlation_heatmap.png\n')
print(corr.round(3))
