"""
Xuất các đồ thị đối sánh chi tiết từng tập dữ liệu bao gồm K-Means và K-Medoids
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from cure_algorithm import CURE, KMedoids

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

os.makedirs('charts', exist_ok=True)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def plot_single_comparison(d_name, X, k, filename, eps=0.25):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(f'SO SÁNH PHÂN CỤM TRÊN TẬP DỮ LIỆU: {d_name.upper()}', fontsize=14, fontweight='bold', color='#103673', y=1.02)
    
    algs = [
        ('CURE', CURE(n_clusters=k, n_representatives=5, shrink_factor=0.4)),
        ('K-Means', KMeans(n_clusters=k, random_state=42, n_init='auto')),
        ('K-Medoids', KMedoids(n_clusters=k, random_state=42)),
        ('DBSCAN', DBSCAN(eps=eps, min_samples=5)),
        ('Hierarchical (Single)', AgglomerativeClustering(n_clusters=k, linkage='single'))
    ]
    
    for idx, (name, model) in enumerate(algs):
        ax = axes[idx]
        if name == 'CURE':
            model.fit(X)
            labels = model.labels_
            reps = model.get_representatives()
            means = model.get_cluster_means()
        elif name == 'K-Medoids':
            labels = model.fit_predict(X)
            reps, means = None, None
        else:
            labels = model.fit_predict(X)
            reps, means = None, None
            
        mask_noise = labels == -1
        if np.any(mask_noise):
            ax.scatter(X[mask_noise, 0], X[mask_noise, 1], c='#BBBBBB', s=16, alpha=0.4, label='Nhiễu')
            
        mask_normal = labels != -1
        pt_colors = [colors[l % len(colors)] for l in labels[mask_normal]]
        ax.scatter(X[mask_normal, 0], X[mask_normal, 1], c=pt_colors, s=20, alpha=0.7)
        
        if reps is not None:
            for rep in reps:
                ax.scatter(rep[:, 0], rep[:, 1], c='black', marker='x', s=55, linewidths=2.2, label='Điểm đại diện sau co')
            ax.scatter(means[:, 0], means[:, 1], c='yellow', edgecolors='black', marker='*', s=140, label='Trọng tâm mean')
        elif name == 'K-Medoids' and model.cluster_centers_ is not None:
            ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='D', s=70, edgecolors='black', label='Medoids')
            
        ax.set_title(name, fontsize=12, fontweight='bold', color='#1f4e79')
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
            
    plt.tight_layout()
    out_path = os.path.join('charts', filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# 1. Moons
X_moons, _ = datasets.make_moons(n_samples=300, noise=0.06, random_state=42)
plot_single_comparison('Two Moons (Trăng khuyết)', X_moons, 2, 'moons_comparison.png', eps=0.25)

# 2. Circles
X_circles, _ = datasets.make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)
plot_single_comparison('Concentric Circles (Vòng tròn đồng tâm)', X_circles, 2, 'circles_comparison.png', eps=0.2)

# 3. Anisotropic
X_blobs, _ = datasets.make_blobs(n_samples=300, cluster_std=[1.0, 1.0, 1.0], random_state=42)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X_blobs, transformation)
plot_single_comparison('Anisotropic (Cụm kéo dài)', X_aniso, 3, 'aniso_comparison.png', eps=0.4)

# 4. Outliers
blobs, _ = datasets.make_blobs(n_samples=260, centers=2, cluster_std=0.8, random_state=42)
rng = np.random.RandomState(42)
outliers = rng.uniform(low=-7, high=7, size=(40, 2))
X_outliers = np.vstack([blobs, outliers])
plot_single_comparison('Blobs with Outliers (Cụm có ngoại lai)', X_outliers, 2, 'outliers_comparison.png', eps=0.5)

print("Export individual charts complete!")
