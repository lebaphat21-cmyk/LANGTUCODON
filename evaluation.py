"""Đo chất lượng trên cùng tập điểm; chỉ số không xác định dùng NaN."""
import numpy as np
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score)


def evaluate_clustering(X, labels, elapsed=0., truth=None):
    X, labels = np.asarray(X), np.asarray(labels)
    valid = labels != -1
    k, n = len(np.unique(labels[valid])), int(valid.sum())
    result = {'Silhouette': np.nan, 'Davies-Bouldin': np.nan, 'Calinski-Harabasz': np.nan,
              'Time': elapsed, 'Clusters': k, 'Noise fraction': float((~valid).mean()),
              'Evaluated points': n, 'ARI': np.nan, 'NMI': np.nan}
    if 2 <= k < n:
        for name, func in [('Silhouette', silhouette_score), ('Davies-Bouldin', davies_bouldin_score),
                           ('Calinski-Harabasz', calinski_harabasz_score)]:
            try:
                result[name] = float(func(X[valid], labels[valid]))
            except ValueError:
                pass
    if truth is not None:
        truth = np.asarray(truth)
        # Cùng mặt nạ nhãn thật cho mọi thuật toán; nhãn dự đoán -1 vẫn được đánh giá.
        known = truth != -1
        if known.sum() > 1:
            result['ARI'] = float(adjusted_rand_score(truth[known], labels[known]))
            result['NMI'] = float(normalized_mutual_info_score(truth[known], labels[known]))
    return result


def format_score(value, digits=4):
    return f'{value:.{digits}f}' if value is not None and np.isfinite(value) else 'N/A'
