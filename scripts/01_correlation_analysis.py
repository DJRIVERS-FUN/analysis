import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
FILE = 'data/381completeanalysis.xlsx'
df = pd.read_excel(FILE)

# Variables of interest
variables = [
    'GPARAW',
    'CMOR',
    'VOFO',
    'Monitoring',
    'Evaluation',
    'Planning',
    'PsychologicalDistress'
]

# Correlation matrix
corr = df[variables].corr()

# Save matrix
corr.to_csv('outputs/correlation_matrix.csv')

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    square=True
)

plt.title('Correlation Matrix: GPA, Executive Function, Metacognition, and Distress')
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=300)

print(corr)
