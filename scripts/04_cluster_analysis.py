import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load dataset
FILE = 'data/381completeanalysis.xlsx'
df = pd.read_excel(FILE)

# Ensure output folders exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Compute executive-function composite
uef_subscales = ['SAS', 'CROE', 'CMOR', 'VOBL', 'MEST', 'DM', 'CROB']
df['EXECFUNC_COMPOSITE'] = df[uef_subscales].mean(axis=1)

# Variables for clustering
cluster_vars = {
    'EXECFUNC_COMPOSITE': 'Executive Function',
    'KOC': 'Metacognitive Knowledge',
    'ROC': 'Metacognitive Regulation',
    'Distress': 'Psychological Distress'
}

analysis_df = df[list(cluster_vars.keys()) + ['GPARAW']].dropna().copy()

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(analysis_df[list(cluster_vars.keys())])

# Determine silhouette scores for k = 2 to 6
silhouette_results = []

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    silhouette_results.append({
        'k': k,
        'silhouette_score': score
    })

silhouette_df = pd.DataFrame(silhouette_results)
silhouette_df.to_csv('outputs/silhouette_scores.csv', index=False)

# Select best k automatically
best_k = silhouette_df.loc[
    silhouette_df['silhouette_score'].idxmax(),
    'k'
]

# Final clustering
kmeans = KMeans(n_clusters=int(best_k), random_state=42, n_init=20)
analysis_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Save cluster means
cluster_summary = analysis_df.groupby('Cluster').agg({
    'EXECFUNC_COMPOSITE': 'mean',
    'KOC': 'mean',
    'ROC': 'mean',
    'Distress': 'mean',
    'GPARAW': ['mean', 'std', 'count']
})

cluster_summary.to_csv('outputs/cluster_summary.csv')

# PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

analysis_df['PC1'] = pca_components[:, 0]
analysis_df['PC2'] = pca_components[:, 1]

# PCA scatterplot
plt.figure(figsize=(7, 6))

scatter = plt.scatter(
    analysis_df['PC1'],
    analysis_df['PC2'],
    c=analysis_df['Cluster']
)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'Cluster Profiles of Academic Self-Regulation (k={best_k})')
plt.tight_layout()

plt.savefig('figures/cluster_pca_plot.png', dpi=300)

# GPA by cluster boxplot
plt.figure(figsize=(7, 5))

analysis_df.boxplot(column='GPARAW', by='Cluster')

plt.title('GPA by Cluster Profile')
plt.suptitle('')
plt.xlabel('Cluster')
plt.ylabel('Raw GPA')
plt.tight_layout()

plt.savefig('figures/gpa_by_cluster.png', dpi=300)

# Save participant cluster assignments
analysis_df.to_csv('outputs/cluster_assignments.csv', index=False)

print('\nCluster analysis complete.')
print(f'Optimal cluster number (silhouette): {best_k}')
print('\nFiles generated:')
print('- outputs/silhouette_scores.csv')
print('- outputs/cluster_summary.csv')
print('- outputs/cluster_assignments.csv')
print('- figures/cluster_pca_plot.png')
print('- figures/gpa_by_cluster.png\n')

print(cluster_summary)
