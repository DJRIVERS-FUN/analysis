import os

import pandas as pd
import matplotlib.pyplot as plt
from semopy import Model, calc_stats
from semopy.inspector import inspect

# Load dataset
FILE = 'data/381completeanalysis.xlsx'
df = pd.read_excel(FILE).copy()

# Ensure output folders exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Compute executive-function composite
uef_subscales = ['SAS', 'CROE', 'CMOR', 'VOBL', 'MEST', 'DM', 'CROB']
execfunc_composite = df[uef_subscales].mean(axis=1).rename('EXECFUNC_COMPOSITE')

# Select variables
analysis_df = pd.concat([
    df[['Distress', 'KOC', 'ROC', 'GPARAW']],
    execfunc_composite
], axis=1).dropna().copy()

# SEM / path model
# Theoretical structure:
# Distress -> Executive Function -> GPA
# Metacognition -> Executive Function -> GPA
# Direct effects also estimated.

model_desc = '''
EXECFUNC_COMPOSITE ~ Distress + KOC + ROC
GPARAW ~ EXECFUNC_COMPOSITE + Distress + KOC + ROC
'''

model = Model(model_desc)
model.fit(analysis_df)

# Parameter estimates
estimates = inspect(model)
estimates.to_csv('outputs/sem_parameter_estimates.csv', index=False)

# Model fit statistics
# semopy versions differ: recent versions expose calc_stats(model) as a function.
stats = calc_stats(model)
fit_df = pd.DataFrame(stats)
fit_df.to_csv('outputs/sem_fit_statistics.csv', index=False)

# Simple path diagram using matplotlib
plt.figure(figsize=(8, 5))
plt.axis('off')

# Node positions
positions = {
    'Distress': (0.1, 0.7),
    'KOC': (0.1, 0.5),
    'ROC': (0.1, 0.3),
    'Executive Function': (0.5, 0.5),
    'GPA': (0.85, 0.5)
}

# Draw nodes
for label, (x, y) in positions.items():
    plt.text(
        x,
        y,
        label,
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.4')
    )

# Draw arrows
arrows = [
    ('Distress', 'Executive Function'),
    ('KOC', 'Executive Function'),
    ('ROC', 'Executive Function'),
    ('Executive Function', 'GPA'),
    ('Distress', 'GPA'),
    ('KOC', 'GPA'),
    ('ROC', 'GPA')
]

for start, end in arrows:
    x1, y1 = positions[start]
    x2, y2 = positions[end]

    plt.annotate(
        '',
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', lw=1.5)
    )

plt.title('SEM / Path Model of GPA Prediction')
plt.tight_layout()
plt.savefig('figures/sem_path_model.png', dpi=300)

print('\nSEM / path model analysis complete.')
print('Files generated:')
print('- outputs/sem_parameter_estimates.csv')
print('- outputs/sem_fit_statistics.csv')
print('- figures/sem_path_model.png\n')

print(estimates)
print('\nFit statistics:')
print(fit_df)
