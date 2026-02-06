import pandas as pd
import numpy as np 
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

reducer = umap.UMAP(n_neighbors=40, min_dist=0.5)

testproj = np.load('/idia/projects/roadtoska/projectG/projectG_supervised_latent_radiogals/outputs/run_closest1.0/embeddings/test_projections.npy')
testlabs = np.load('/idia/projects/roadtoska/projectG/projectG_supervised_latent_radiogals/outputs/run_closest1.0/embeddings/test_labels.npy')[:,:2]

# testlabs shape: (N, 5) with 0/1
combo_strings = ["".join(map(str, row.astype(int))) for row in testlabs]

# Unique combinations that ACTUALLY appear
unique_combos = sorted(set(combo_strings))

# Map combo → integer id
combo_to_id = {c: i for i, c in enumerate(unique_combos)}
combo_ids = np.array([combo_to_id[c] for c in combo_strings])


scaled_proj_data = StandardScaler().fit_transform(testproj)

embedding = reducer.fit_transform(scaled_proj_data)
print(embedding.shape)

num_classes = len(unique_combos)
palette = sns.color_palette("husl", num_classes)  # good separation even for many classes

from matplotlib.lines import Line2D

def pretty_label(c):
    return f"[{c[0]}, {c[1]}]"

handles = [
    Line2D([0], [0],
           marker='o',
           color='w',
           label=pretty_label(combo),
           markerfacecolor=palette[idx],
           markersize=8)
    for combo, idx in combo_to_id.items()
]


plt.figure(figsize=(8,8))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[palette[i] for i in combo_ids])
plt.legend(handles=handles, title="Tag Combo", loc="best")
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the proj dataset', fontsize=24)
plt.savefig('umap_test.png')

