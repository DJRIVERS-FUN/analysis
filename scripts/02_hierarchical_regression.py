import os

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load dataset
FILE = 'data/381completeanalysis.xlsx'
df = pd.read_excel(FILE)

# Ensure output folders exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Variables
outcome = 'GPARAW'

step1_vars = ['YEAR', 'Sex', 'Age']
step2_vars = [
    'Distress',
    'METAKNOW',
    'METAREG',
    'EXECFUNC'
]

# Clean dataset
analysis_df = df[[outcome] + step1_vars + step2_vars].dropna()

# Standardize predictors for beta coefficients
for col in step1_vars + step2_vars:
    analysis_df[col] = (
        analysis_df[col] - analysis_df[col].mean()
    ) / analysis_df[col].std()

# Step 1 regression
X1 = sm.add_constant(analysis_df[step1_vars])
y = analysis_df[outcome]

model1 = sm.OLS(y, X1).fit()

# Step 2 regression
X2 = sm.add_constant(analysis_df[step1_vars + step2_vars])
model2 = sm.OLS(y, X2).fit()

# Save summaries
with open('outputs/hierarchical_regression_summary.txt', 'w') as f:
    f.write('STEP 1 MODEL\n')
    f.write(model1.summary().as_text())
    f.write('\n\nSTEP 2 MODEL\n')
    f.write(model2.summary().as_text())

# Create coefficient dataframe
coef_df = pd.DataFrame({
    'Predictor': model2.params.index,
    'Beta': model2.params.values,
    'CI_lower': model2.conf_int()[0].values,
    'CI_upper': model2.conf_int()[1].values
})

# Remove intercept
coef_df = coef_df[coef_df['Predictor'] != 'const']

# Rename for readability
rename_dict = {
    'YEAR': 'Year of Study',
    'Sex': 'Sex',
    'Age': 'Age',
    'Distress': 'Psychological Distress',
    'METAKNOW': 'Metacognitive Knowledge',
    'METAREG': 'Metacognitive Regulation',
    'EXECFUNC': 'Executive Function'
}

coef_df['Predictor'] = coef_df['Predictor'].replace(rename_dict)

# Forest plot
plt.figure(figsize=(8, 5))

plt.errorbar(
    coef_df['Beta'],
    coef_df['Predictor'],
    xerr=[
        coef_df['Beta'] - coef_df['CI_lower'],
        coef_df['CI_upper'] - coef_df['Beta']
    ],
    fmt='o'
)

plt.axvline(0, linestyle='--')
plt.xlabel('Standardized Beta Coefficient')
plt.title('Hierarchical Regression Predicting GPA')
plt.tight_layout()

plt.savefig('figures/regression_forest_plot.png', dpi=300)

# Save coefficient table
coef_df.to_csv('outputs/regression_coefficients.csv', index=False)

print('\nHierarchical regression complete.')
print('Regression summary saved to outputs/hierarchical_regression_summary.txt')
print('Coefficient table saved to outputs/regression_coefficients.csv')
print('Forest plot saved to figures/regression_forest_plot.png\n')

print(model2.summary())
