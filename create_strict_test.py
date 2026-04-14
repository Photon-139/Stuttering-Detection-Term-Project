import nbformat as nbf
import os, glob

nb = nbf.v4.new_notebook()

m1 = nbf.v4.new_markdown_cell("# Strict Labelling Analysis & Hybrid Strategy\nWe are testing the impact of 'Strict Agreement' filtering on the disfluent dataset while using the TA's 'non_stutter' dataset as the fluent reference.")

c1 = nbf.v4.new_code_cell("""
import os, glob, numpy as np, matplotlib.pyplot as plt
from src.data import DataManager
from sklearn.manifold import TSNE

manager = DataManager()
csv_paths = ['Stuttering Events in Podcasts Dataset/SEP-28k_labels.csv', 'Stuttering Events in Podcasts Dataset/SEP-28k_episodes.csv']
strict_dict = manager.generate_label_dict(csv_paths, strict=True)
""")

c2 = nbf.v4.new_code_cell("""
# APPROACH: HYBRID STRICT
# 1. Take all TA 'non_stutter' samples
f_total = glob.glob('non_stutter/*.npy')
X_f = np.vstack([np.load(f) for f in f_total])
y_f = np.zeros(len(f_total))

# 2. Take only STUTTER samples from 'disfluent' with 3/3 agreement
d_total = glob.glob('data/features/disfluent/*.npy')
d_files = [f for f in d_total if os.path.splitext(os.path.basename(f))[0] in strict_dict]
X_d = np.vstack([np.load(f) for f in d_files])
y_d = np.ones(len(d_files))

X = np.vstack([X_f, X_d])
y = np.concatenate([y_f, y_d])

print(f"Hybrid Strict Dataset Ready.")
print(f"Fluent (TA): {len(y_f)}")
print(f"Disfluent (Strict 3/3): {len(y_d)}")
""")

c3 = nbf.v4.new_code_cell("""
print("Running t-SNE visualization...")
tsne = TSNE(n_components=2, init='pca', random_state=42)
idx = np.random.choice(len(X), min(2000, len(X)), replace=False)
X_2d = tsne.fit_transform(X[idx])
y_2d = y[idx]

plt.figure(figsize=(10, 8))
plt.scatter(X_2d[y_2d==0, 0], X_2d[y_2d==0, 1], c='lightblue', edgecolors='k', alpha=0.6, label='Fluent (TA)')
plt.scatter(X_2d[y_2d==1, 0], X_2d[y_2d==1, 1], c='darkred', edgecolors='k', alpha=0.6, label='Stutter (Strict)')
plt.title("Hybrid Strict Manifold")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
""")

nb['cells'] = [m1, c1, c2, c3]

with open('Strict_Labelling_Tests.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created.")
