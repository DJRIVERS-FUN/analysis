import os

import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt

# Load dataset
FILE = 'data/381completeanalysis.xlsx'
df = pd.read_excel(FILE)

# Ensure output folders exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Compute executive-function composite from UEF subscales
uef_subscales = ['SAS', 'CROE', 'CMOR', 'VOBL', 'MEST', 'DM', 'CROB']
df['EXECFUNC_COMPOSITE'] = df[uef_subscales].mean(axis=1)

# Define key predictors and controls
outcome = 'GPARAW'
key_predictors = {
    'EXECFUNC_COMPOSITE': 'Executive Function',
    'CMOR': 'Conscious Monitoring of Responsibilities',
    'VOBL': 'Verification of Fulfilment of Objectives',
    'KOC': 'Metacognitive Knowledge',
    'ROC': 'Metacognitive Regulation',
    'Distress': 'Psychological Distress'
}

controls = ['YEAR', 'Sex', 'Age']
construct_controls = ['KOC', 'ROC', 'Distress']


def get_first_existing(row, possible_names, default=''):
    """Return a value from the first matching column name."""
    for name in possible_names:
        if name in row.index:
            return row[name]
    return default


results = []

for predictor, label in key_predictors.items():
    # For the full EF composite, control for metacognition, distress, and demographics.
    # For other predictors, control for demographics only to avoid over-control among overlapping subscales.
    if predictor == 'EXECFUNC_COMPOSITE':
        covars = controls + construct_controls
    else:
        covars = controls

    temp_vars = [outcome, predictor] + covars
    temp_df = df[temp_vars].dropna()

    pc = pg.partial_corr(
        data=temp_df,
        x=predictor,
        y=outcome,
        covar=covars,
        method='pearson'
    )

    row = pc.iloc[0]

    results.append({
        'Predictor': label,
        'Partial_r': get_first_existing(row, ['r', 'R']),
        'CI95': get_first_existing(row, ['CI95%', 'CI95', 'CI95%_r', 'ci95']),
        'p_value': get_first_existing(row, ['p-val', 'p_val', 'pvalue', 'p-value', 'p']),
        'n': get_first_existing(row, ['n', 'N']),
        'Controls': ', '.join(covars)
    })

results_df = pd.DataFrame(results)
results_df.to_csv('outputs/partial_correlations.csv', index=False)

# Plot partial correlations
plot_df = results_df.copy().sort_values('Partial_r')

plt.figure(figsize=(8, 5))
plt.barh(plot_df['Predictor'], plot_df['Partial_r'])
plt.axvline(0, linestyle='--', linewidth=1)
plt.xlabel('Partial correlation with GPA')
plt.title('Partial Correlations Predicting GPA')
plt.tight_layout()
plt.savefig('figures/partial_correlations.png', dpi=300)

print('\nPartial correlation analysis complete.')
print('Results saved to outputs/partial_correlations.csv')
print('Figure saved to figures/partial_correlations.png\n')
print(results_df.round(3))
