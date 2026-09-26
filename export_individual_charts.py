"""Xuất hình đối sánh cùng cấu hình với benchmark; nhãn thật để tham chiếu."""
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from data_pipeline import DATASETS, synthetic_data
from experiments import ALGORITHMS, fit_model
from benchmark_comparison import plot_example


def export_charts():
    out=Path('charts'); out.mkdir(exist_ok=True)
    for name, filename in zip(DATASETS[1:], ['moons_comparison.png','circles_comparison.png','aniso_comparison.png','outliers_comparison.png']):
        raw, truth, k = synthetic_data(name, 200, 42)
        X=StandardScaler().fit_transform(raw)
        models={a:fit_model(a,X,k,4,.4,42,history=False)[0] for a in ALGORITHMS+['DBSCAN']}
        plot_example(name,X,truth,models,out/filename)


if __name__ == '__main__':
    export_charts()
