"""
Thực nghiệm So sánh và Đánh giá thuật toán CURE với K-Means, K-Medoids, DBSCAN, Hierarchical
Môn: Khai phá dữ liệu - ĐH Công Thương TP.HCM (HUIT)
"""
import os
import sys
import time

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from cure_algorithm import CURE, KMedoids

os.makedirs('charts', exist_ok=True)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def generate_datasets(n_samples=300, random_state=42):
    moons, _ = datasets.make_moons(n_samples=n_samples, noise=0.06, random_state=random_state)
    circles, _ = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
    X_blobs, _ = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 1.0, 1.0], random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    aniso = np.dot(X_blobs, transformation)
    blobs, _ = datasets.make_blobs(n_samples=n_samples - 40, centers=2, cluster_std=0.8, random_state=random_state)
    rng = np.random.RandomState(random_state)
    outliers = rng.uniform(low=-7, high=7, size=(40, 2))
    outliers_data = np.vstack([blobs, outliers])
    
    return [
        ('1. Two Moons (Trăng khuyết)', moons, 2),
        ('2. Concentric Circles (Vòng tròn đồng tâm)', circles, 2),
        ('3. Anisotropic (Cụm kéo dài)', aniso, 3),
        ('4. Blobs with Outliers (Cụm có ngoại lai)', outliers_data, 2)
    ]

def evaluate_clustering(X, labels):
    unique_labels = set(labels)
    valid_mask = labels != -1
    n_clusters = len(set(labels[valid_mask]))
    
    if n_clusters < 2 or len(unique_labels) == 1:
        return {'silhouette': -1.0, 'davies_bouldin': 99.0, 'calinski': 0.0, 'n_clusters': n_clusters}
    
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    sil = silhouette_score(X_valid, labels_valid)
    db = davies_bouldin_score(X_valid, labels_valid)
    ch = calinski_harabasz_score(X_valid, labels_valid)
    return {'silhouette': sil, 'davies_bouldin': db, 'calinski': ch, 'n_clusters': n_clusters}

def run_benchmark():
    print("=== BẮT ĐẦU THỰC NGHIỆM ĐỐI SÁNH: CURE VS K-MEANS VS K-MEDOIDS VS DBSCAN VS HIERARCHICAL ===")
    dataset_list = generate_datasets()
    
    algorithms = [
        ('CURE', lambda k: CURE(n_clusters=k, n_representatives=5, shrink_factor=0.4)),
        ('K-Means', lambda k: KMeans(n_clusters=k, random_state=42, n_init='auto')),
        ('K-Medoids', lambda k: KMedoids(n_clusters=k, random_state=42)),
        ('DBSCAN', lambda k: DBSCAN(eps=0.25 if k==2 else 0.4, min_samples=5)),
        ('Hierarchical (Single)', lambda k: AgglomerativeClustering(n_clusters=k, linkage='single'))
    ]
    
    results = []
    fig, axes = plt.subplots(len(dataset_list), len(algorithms), figsize=(22, 16))
    fig.suptitle('SO SÁNH ĐỐI ĐẦU: CURE VS K-MEANS VS K-MEDOIDS VS DBSCAN VS HIERARCHICAL\n(Trường ĐH Công Thương TP.HCM - HUIT)', 
                 fontsize=16, fontweight='bold', color='#103673', y=0.995)
    
    colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    
    for row_idx, (d_name, X, k) in enumerate(dataset_list):
        print(f"\n--- Đang xử lý tập dữ liệu: {d_name} ---")
        for col_idx, (alg_name, alg_factory) in enumerate(algorithms):
            model = alg_factory(k)
            t0 = time.time()
            if alg_name == 'CURE':
                model.fit(X)
                labels = model.labels_
                reps = model.get_representatives()
            elif alg_name == 'K-Medoids':
                labels = model.fit_predict(X)
                reps = None
            elif alg_name == 'DBSCAN':
                labels = model.fit_predict(X)
                reps = None
            else:
                labels = model.fit_predict(X)
                reps = None
            runtime = time.time() - t0
            
            metrics = evaluate_clustering(X, labels)
            
            results.append({
                'Tập dữ liệu': d_name,
                'Thuật toán': alg_name,
                'Số cụm tìm được': metrics['n_clusters'],
                'Silhouette Score': round(metrics['silhouette'], 4),
                'Davies-Bouldin': round(metrics['davies_bouldin'], 4),
                'Calinski-Harabasz': round(metrics['calinski'], 1),
                'Thời gian (giây)': round(runtime, 4)
            })
            
            ax = axes[row_idx, col_idx]
            mask_noise = labels == -1
            if np.any(mask_noise):
                ax.scatter(X[mask_noise, 0], X[mask_noise, 1], c='#CCCCCC', s=15, alpha=0.5, label='Nhiễu')
                
            mask_normal = labels != -1
            point_colors = [colors[l % len(colors)] for l in labels[mask_normal]]
            ax.scatter(X[mask_normal, 0], X[mask_normal, 1], c=point_colors, s=18, alpha=0.7)
            
            if alg_name == 'CURE' and reps is not None:
                for rep in reps:
                    ax.scatter(rep[:, 0], rep[:, 1], c='black', marker='x', s=45, linewidths=2, zorder=5)
                means = model.get_cluster_means()
                ax.scatter(means[:, 0], means[:, 1], c='yellow', edgecolors='black', marker='*', s=120, zorder=6)
            elif alg_name == 'K-Medoids' and model.cluster_centers_ is not None:
                ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='D', s=70, edgecolors='black', zorder=6, label='Medoids')
                
            if row_idx == 0:
                ax.set_title(alg_name, fontsize=13, fontweight='bold', color='#103673', pad=10)
                
            if col_idx == 0:
                ax.set_ylabel(d_name, fontsize=10, fontweight='bold', color='#1f4e79')
                
            sil_str = f"Silhouette: {metrics['silhouette']:.3f}" if metrics['silhouette'] != -1.0 else "N/A"
            ax.text(0.03, 0.05, f"{sil_str}\nTime: {runtime:.3f}s", 
                    transform=ax.transAxes, fontsize=8.5, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#AAAAAA'))
            
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    overview_path = os.path.join('charts', 'cure_vs_others_all_datasets.png')
    plt.savefig(overview_path, dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ tổng thể: {overview_path}")
    
    df_results = pd.DataFrame(results)
    csv_path = os.path.join('charts', 'metrics_comparison_table.csv')
    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"-> Đã lưu bảng chỉ số ra CSV: {csv_path}")
    
    # Biểu đồ cột Silhouette
    fig_bar, ax_bar = plt.subplots(figsize=(13, 6))
    pivot_sil = df_results.pivot(index='Tập dữ liệu', columns='Thuật toán', values='Silhouette Score')
    pivot_sil.plot(kind='bar', ax=ax_bar, colormap='viridis', width=0.8)
    ax_bar.set_title('SO SÁNH CHỈ SỐ SILHOUETTE SCORE (CÀNG CAO CÀNG TỐT)', fontsize=13, fontweight='bold', color='#103673')
    ax_bar.set_ylabel('Silhouette Score')
    ax_bar.set_xlabel('')
    ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=15, ha='right')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='Thuật toán', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    bar_path = os.path.join('charts', 'silhouette_comparison_barchart.png')
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ cột Silhouette: {bar_path}")
    
    return df_results

if __name__ == '__main__':
    df = run_benchmark()
    print("\n=== KẾT QUẢ ĐỐI SÁNH TỔNG HỢP ===")
    print(df.to_string(index=False))
