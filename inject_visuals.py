import json
import glob
import nbformat as nbf

notebooks = [
    'KNN_Distance_Analysis.ipynb', 
    'SVM_Kernel_Deep_Dive.ipynb', 
    'Neural_Network_Deep_Dive.ipynb', 
    'Tree_Ensemble_Deep_Dive.ipynb', 
    'Linear_Models_Deep_Dive.ipynb', 
    'Probabilistic_Models_Deep_Dive.ipynb'
]

legend_injection_code = '''
# --- UNIFIED LEGEND ---
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements_20 = [
    Line2D([0], [0], color='black', lw=3, label='Decision Boundary / Topography'),
    Line2D([0], [0], color='darkred', lw=2, label='Stutter Prediction Confidence Region'),
    Line2D([0], [0], color='darkblue', lw=2, label='Fluent Prediction Confidence Region'),
    Line2D([0], [0], marker='o', color='w', label='Actual Test Data Point', markerfacecolor='gray', markeredgecolor='black', markersize=9, alpha=0.7)
]
plt.legend(handles=legend_elements_20, loc='lower center', ncol=2, frameon=True, fontsize=11, bbox_to_anchor=(0.5, -0.3))
plt.tight_layout()
plt.show()
'''

tsne_code = '''\
print("Running highly-detailed t-SNE dimensionality reduction to visualize manifold overlap...")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 1. Limit samples for TSNE performance
np.random.seed(42)
VIS_LIMIT = 2000
# Neural network prepends tensor logic, fallbacks to raw splits
try:
    X_target = X_train_final
    y_target = y_train_bal
except:
    X_target = X_test_final
    y_target = y_test

if len(X_target) > VIS_LIMIT:
    idx = np.random.choice(len(X_target), VIS_LIMIT, replace=False)
    X_vis = X_target[idx]
    y_vis = y_target[idx]
else:
    X_vis, y_vis = X_target, y_target

# 2. Run TSNE
tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X_vis)

# 3. Plot
plt.figure(figsize=(10, 8))
plt.scatter(X_2d[y_vis==0, 0], X_2d[y_vis==0, 1], c='lightblue', edgecolors='k', alpha=0.6, s=40)
plt.scatter(X_2d[y_vis==1, 0], X_2d[y_vis==1, 1], c='darkred', edgecolors='k', alpha=0.6, s=40)
plt.title(f"Model Feature Manifold - t-SNE Projection\\n(Validating Clean Separation with new TA Data)", fontsize=14)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(alpha=0.3)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Fluent/Non-Stutter', markerfacecolor='lightblue', markeredgecolor='k', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Stutter', markerfacecolor='darkred', markeredgecolor='k', markersize=10)
]
plt.legend(handles=legend_elements, loc='best')
plt.tight_layout()
plt.show()
'''


for nb_path in notebooks:
    with open(nb_path, 'r') as f:
        nb = json.load(f)
        
    modified = False
    has_tsne = False
    
    # Remove existing flawed t-sne cells
    new_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'tsne' in src.lower() or 't-sne' in src.lower():
                continue # Skip old tsne
                
            if 'plt.show()' in src and ('contourf' in src or 'decision_function' in src or 'plot_decision_regions' in src or 'Decision Boundary' in src):
                if 'UNIFIED LEGEND' not in src:
                    # Inject legend
                    new_src = src.replace('plt.show()', legend_injection_code)
                    cell['source'] = new_src.splitlines(True)
                    modified = True
                    
        new_cells.append(cell)
        
    nb['cells'] = new_cells
    
    # Append the new perfect TSNE cell to the END of the notebook!
    tsne_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": tsne_code.splitlines(True)
    }
    nb['cells'].append(tsne_cell)
    modified = True
    
    if modified:
        with open(nb_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f'Successfully injected Aesthetics and t-SNE into {nb_path}')
