import os

import pandas as pd
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

# The dataset contains metacognition composites as KOC and ROC.
# KOC = metacognitive knowledge; ROC = metacognitive regulation.
# Executive functioning is computed as the mean of the seven UEF subscales.
uef_subscales = ['SAS', 'CROE', 'CMOR', 'VOBL', 'MEST', 'DM', 'CROB']
df['EXECFUNC_COMPOSITE'] = df[uef_subscales].mean(axis=1)

step2_vars = [
    'Distress',
    'KOC',
    'ROC',
    'EXECFUNC_COMPOSITE'
]

# Clean dataset
analysis_df = df[[outcome] + step1_vars + step2_vars].dropna().copy()

# Standardize predictors for beta coefficients
for col in step1_vars + step2_vars:
    analysis_df[col] = (
        analysis_df[col] - analysis_df[col].mean()
    ) / analysis_df[col].std()

# Step 1 regression
y = analysis_df[outcome]
X1 = sm.add_constant(analysis_df[step1_vars])
model1 = sm.OLS(y, X1).fit()

# Step 2 regression
X2 = sm.add_constant(analysis_df[step1_vars + step2_vars])
model2 = sm.OLS(y, X2).fit()

# Model change statistics
r2_step1 = model1.rsquared
r2_step2 = model2.rsquared
delta_r2 = r2_step2 - r2_step1

# Save summaries
with open('outputs/hierarchical_regression_summary.txt', 'w') as f:
    f.write('STEP 1 MODEL\n')
    f.write(model1.summary().as_text())
    f.write('\n\nSTEP 2 MODEL\n')
    f.write(model2.summary().as_text())
    f.write('\n\nMODEL COMPARISON\n')
    f.write(f'Step 1 R-squared: {r2_step1:.4f}\n')
    f.write(f'Step 2 R-squared: {r2_step2:.4f}\n')
    f.write(f'Delta R-squared: {delta_r2:.4f}\n')

# Create coefficient dataframe
conf = model2.conf_int()
coef_df = pd.DataFrame({
    'Predictor': model2.params.index,
    'Beta': model2.params.values,
    'CI_lower': conf[0].values,
    'CI_upper': conf[1].values,
    'p_value': model2.pvalues.values
})

# Remove intercept
coef_df = coef_df[coef_df['Predictor'] != 'const'].copy()

# Rename for readability
rename_dict = {
    'YEAR': 'Year of Study',
    'Sex': 'Sex',
    'Age': 'Age',
    'Distress': 'Psychological Distress',
    'KOC': 'Metacognitive Knowledge',
    'ROC': 'Metacognitive Regulation',
    'EXECFUNC_COMPOSITE': 'Executive Function'
}

coef_df['Predictor'] = coef_df['Predictor'].replace(rename_dict)

# Save coefficient table
coef_df.to_csv('outputs/regression_coefficients.csv', index=False)

# Forest plot
plt.figure(figsize=(8, 5))
plt.errorbar(
    coef_df['Beta'],
    coef_df['Predictor'],
    xerr=[
        coef_df['Beta'] - coef_df['CI_lower'],
        coef_df['CI_upper'] - coef_df['Beta']
    ],
    fmt='o',
    capsize=4
)

plt.axvline(0, linestyle='--', linewidth=1)
plt.xlabel('Standardized Beta Coefficient')
plt.title('Hierarchical Regression Predicting GPA')
plt.tight_layout()
plt.savefig('figures/regression_forest_plot.png', dpi=300)

print('\nHierarchical regression complete.')
print('Regression summary saved to outputs/hierarchical_regression_summary.txt')
print('Coefficient table saved to outputs/regression_coefficients.csv')
print('Forest plot saved to figures/regression_forest_plot.png\n')
print(f'Step 1 R-squared: {r2_step1:.4f}')
print(f'Step 2 R-squared: {r2_step2:.4f}')
print(f'Delta R-squared: {delta_r2:.4f}\n')
print(model2.summary())
