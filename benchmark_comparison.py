"""Thực nghiệm lặp trên 4 bộ dữ liệu, xuất CSV, cấu hình và biểu đồ.

python benchmark_comparison.py --seeds 11 22 33 44 55 --repeats 3 --samples 200
"""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_pipeline import DATASETS, synthetic_data
from experiments import ALGORITHMS, fit_model
from evaluation import evaluate_clustering


def generate_datasets(n_samples=300, random_state=42):
    return [(name, *synthetic_data(name, n_samples, random_state)) for name in DATASETS[1:]]


def plot_example(name, X, truth, models, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(19, 7))
    entries = [('Nhãn thật', truth)] + [(name, m.labels_) for name, m in models.items()]
    for ax, (label, labels) in zip(axes.flat, entries):
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=9)
        ax.set_title(label, fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(entries):]:
        ax.set_visible(False)
    fig.suptitle(name + ' — một lượt minh họa (xem CSV để đánh giá nhiều seed)')
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run_benchmark(seeds=(11, 22, 33, 44, 55), repeats=3, n_samples=200,
                  standardize=True, output='benchmark_results', c=4, alpha=.4, eps=.25):
    if n_samples < 20 or repeats < 1 or not seeds:
        raise ValueError('Cần ít nhất 20 điểm, 1 lần lặp và 1 seed.')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_index, name in enumerate(DATASETS[1:]):
        for seed in seeds:
            raw, truth, k = synthetic_data(name, n_samples, seed)
            X = StandardScaler().fit_transform(raw) if standardize else raw
            # Warm-up riêng từng thuật toán, không đưa vào thời gian báo cáo.
            for algorithm in ALGORITHMS + ['DBSCAN']:
                fit_model(algorithm, X, k, c, alpha, seed, history=False, eps=eps)
            example = {}
            for repeat in range(repeats):
                # Đảo thứ tự có seed để giảm thiên lệch nhiệt/tải theo vị trí chạy.
                order = np.random.RandomState(seed + repeat).permutation(ALGORITHMS + ['DBSCAN'])
                for algorithm in order:
                    model, elapsed = fit_model(algorithm, X, k, c, alpha, seed, history=False, eps=eps)
                    metrics = evaluate_clustering(X, model.labels_, elapsed, truth)
                    rows.append({'Dataset': name, 'Algorithm': algorithm, 'Seed': seed,
                                 'Repeat': repeat+1, 'N': len(X), 'k_target': k, **metrics})
                    if repeat == 0:
                        example[algorithm] = model
            if seed == seeds[0]:
                plot_example(name, X, truth, {n: example[n] for n in ALGORITHMS + ['DBSCAN']},
                             out / f'dataset_{dataset_index+1}.png')
            print(f'Completed {name}, seed={seed}', flush=True)
    raw_results = pd.DataFrame(rows)
    raw_results.to_csv(out / 'runs.csv', index=False, encoding='utf-8-sig')
    metrics = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'ARI', 'NMI',
               'Time', 'Clusters', 'Noise fraction', 'Evaluated points']
    # Chất lượng: trung bình mỗi seed trước, không coi lần lặp thời gian là mẫu độc lập.
    per_seed = raw_results.groupby(['Dataset', 'Algorithm', 'Seed'])[metrics].mean()
    summary = per_seed.groupby(['Dataset', 'Algorithm']).agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / 'summary.csv', index=False, encoding='utf-8-sig')
    timing = raw_results.groupby(['Dataset', 'Algorithm']).Time.agg(['mean', 'std', 'min', 'max', 'count'])
    timing.to_csv(out / 'timings.csv', encoding='utf-8-sig')
    config = dict(seeds=list(seeds), repeats=repeats, samples=n_samples, standardize=standardize,
                  c=c, alpha=alpha, dbscan_eps=eps, dbscan_min_samples=5, kmeans_n_init=10,
                  blas_threads=1, warmup='one fit per algorithm and dataset/seed',
                  timing='perf_counter, fit only, no metrics/preprocessing/history',
                  quality='ARI/NMI on fixed ground-truth non-noise mask; predicted noise retained',
                  internal_metrics='Exclude predicted -1; report coverage and noise fraction',
                  summary='Mean/std across seed means. Timing details across all repeats in timings.csv.',
                  limitations='Fixed parameters, not an optimal-parameter ranking; no CURE partitioning/outlier removal.',
                  versions={name: version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'pandas', 'matplotlib']})
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11,22,33,44,55])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--no-standardize', action='store_true')
    parser.add_argument('--output', default='benchmark_results')
    args = parser.parse_args()
    run_benchmark(args.seeds, args.repeats, args.samples, not args.no_standardize, args.output)
