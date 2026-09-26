"""Cùng cấu hình thuật toán cho ứng dụng và benchmark."""
from time import perf_counter
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from threadpoolctl import threadpool_limits
from cure_algorithm import CURE, KMedoids, DIANA
from evaluation import evaluate_clustering

ALGORITHMS = ['CURE', 'K-Means', 'K-Medoids (PAM)', 'AGNES (single)',
              'AGNES (complete)', 'AGNES (average)', 'AGNES (ward)', 'DIANA']


def make_model(name, k=2, c=4, alpha=.4, seed=42, history=True, eps=.25):
    if name == 'CURE':
        return CURE(k, c, alpha, random_state=seed, record_history=history)
    if name == 'K-Means':
        return KMeans(n_clusters=k, n_init=10, random_state=seed)
    if name == 'K-Medoids (PAM)':
        return KMedoids(k, random_state=seed)
    if name == 'DIANA':
        return DIANA(k)
    if name.startswith('AGNES ('):
        return AgglomerativeClustering(n_clusters=k, linkage=name[7:-1])
    if name == 'DBSCAN':
        return DBSCAN(eps=eps, min_samples=5)
    raise ValueError(f'Thuật toán không hợp lệ: {name}')


def fit_model(name, X, k=2, c=4, alpha=.4, seed=42, history=True, eps=.25):
    model = make_model(name, k, c, alpha, seed, history, eps)
    # Một luồng tính toán giúp giảm nhiễu do lịch chạy BLAS giữa các phép đo.
    with threadpool_limits(limits=1):
        start = perf_counter()
        model.fit(X)
        elapsed = perf_counter() - start
    return model, elapsed


def run_experiment(name, X, k=2, c=4, alpha=.4, seed=42, truth=None):
    model, elapsed = fit_model(name, X, k, c, alpha, seed)
    return model, evaluate_clustering(X, model.labels_, elapsed, truth)
