import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.data import DataManager

manager = DataManager()
label_dict = {'fluent': 0, 'disfluent': 1}

print('Loading Old Data...')
old_X, old_y = manager.load_from_folders(
    'data/features/fluent', 
    'data/features/disfluent', 
    limit=2000, 
    label_dict=label_dict
)

print('Loading New Data...')
new_X, new_y = manager.load_from_folders(
    'non_stutter', 
    'data/features/disfluent', 
    limit=2000, 
    label_dict={'non_stutter': 0, 'disfluent': 1}
)

print(f'Old Data Shape: {old_X.shape}, New Data Shape: {new_X.shape}')

def plot_tsne(X, y, title):
    np.random.seed(42)
    VIS_LIMIT = 2000
    if len(X) > VIS_LIMIT:
        idx = np.random.choice(len(X), VIS_LIMIT, replace=False)
        X_vis = X[idx]
        y_vis = y[idx]
    else:
        X_vis, y_vis = X, y
        
    tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30, n_jobs=-1)
    X_2d = tsne.fit_transform(X_vis)
    
    plt.scatter(X_2d[y_vis==0, 0], X_2d[y_vis==0, 1], c='lightblue', edgecolors='k', alpha=0.6, label='Fluent (Class 0)')
    plt.scatter(X_2d[y_vis==1, 0], X_2d[y_vis==1, 1], c='darkred', edgecolors='k', alpha=0.6, label='Stutter (Class 1)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
plot_tsne(old_X, old_y, 'Original Dataset t-SNE\n(Heavy Overlap Manifold)')

plt.subplot(1, 2, 2)
plot_tsne(new_X, new_y, 'New TA Dataset t-SNE\n(\'non_stutter\' vs Original \'disfluent\')')

plt.tight_layout()
plt.savefig('overlap_comparison.png', dpi=300)
print('Plot saved as overlap_comparison.png')
