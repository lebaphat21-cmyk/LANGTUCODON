"""Dữ liệu thực nghiệm: tách dữ liệu mô hình, dữ liệu hiển thị và nhãn thật."""
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATASETS = ['Khách hàng (Data/Test.csv)', 'Two Moons', 'Concentric Circles',
            'Anisotropic Blobs', 'Blobs with Outliers']
FEATURES = {
    'Tuổi vs Điểm chi tiêu': ('Age', 'Spending_Score_Num'),
    'Tuổi vs Kinh nghiệm': ('Age', 'Work_Experience'),
    'Tuổi vs Quy mô gia đình': ('Age', 'Family_Size'),
    'Kinh nghiệm vs Quy mô gia đình': ('Work_Experience', 'Family_Size'),
    'PCA 2D (4 thuộc tính)': None,
}
AXES = {'Age': 'Tuổi (Age)', 'Spending_Score_Num': 'Điểm chi tiêu (1:Low, 2:Average, 3:High)',
        'Work_Experience': 'Kinh nghiệm (năm)', 'Family_Size': 'Quy mô gia đình'}
TOY_POINTS = np.array([[1, 2], [2, 3], [2, 1], [8, 7], [9, 8], [8, 9]], dtype=float)


def synthetic_data(name, n=300, seed=42):
    if name == 'Two Moons':
        return (*datasets.make_moons(n_samples=n, noise=.06, random_state=seed), 2)
    if name == 'Concentric Circles':
        return (*datasets.make_circles(n_samples=n, factor=.5, noise=.05, random_state=seed), 2)
    if name == 'Anisotropic Blobs':
        X, y = datasets.make_blobs(n_samples=n, centers=3, cluster_std=.9, random_state=seed)
        return X @ np.array([[.6, -.6], [-.4, .8]]), y, 3
    if name == 'Blobs with Outliers':
        noise_n = max(1, round(n * .13))
        X, y = datasets.make_blobs(n_samples=n-noise_n, centers=2, cluster_std=.8, random_state=seed)
        noise = np.random.RandomState(seed).uniform(-7, 7, (noise_n, 2))
        return np.vstack((X, noise)), np.concatenate((y, np.full(noise_n, -1))), 2
    raise ValueError('Tập dữ liệu không được hỗ trợ.')


def customer_data(source, pair, n, seed, standardize=True):
    try:
        df = pd.read_csv(BytesIO(source) if isinstance(source, bytes) else Path(source))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError('Không đọc được CSV UTF-8 hợp lệ.') from exc
    required = ['Age', 'Spending_Score', 'Work_Experience', 'Family_Size']
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError('CSV thiếu cột: ' + ', '.join(missing))
    if len(df) < 2:
        raise ValueError('CSV cần ít nhất 2 dòng.')
    if pair not in FEATURES:
        raise ValueError('Cặp thuộc tính không hợp lệ.')
    sample = df.sample(n=min(n, len(df)), random_state=seed).copy()
    notes = []
    for column in ['Age', 'Work_Experience', 'Family_Size']:
        values = pd.to_numeric(sample[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() == 0:
            raise ValueError(f'Cột {column} không có giá trị số hợp lệ trong mẫu.')
        if values.isna().any():
            notes.append(f'{column}: điền trung vị cho {values.isna().sum()} giá trị thiếu/không hợp lệ.')
        sample[column] = values.fillna(values.median())
    spending = sample.Spending_Score.astype('string').str.strip().str.lower().map({'low': 1, 'average': 2, 'high': 3})
    if spending.notna().sum() == 0:
        raise ValueError('Spending_Score phải có giá trị Low, Average hoặc High.')
    if spending.isna().any():
        notes.append(f'Chi tiêu: điền giá trị phổ biến cho {spending.isna().sum()} giá trị thiếu/không hợp lệ.')
    sample['Spending_Score_Num'] = spending.fillna(spending.mode().iloc[0]).astype(float)
    columns = FEATURES[pair]
    transformer = None
    if columns is None:
        raw = sample[['Age', 'Spending_Score_Num', 'Work_Experience', 'Family_Size']].to_numpy(float)
        scaled = StandardScaler().fit_transform(raw)
        if not np.any(scaled):
            raise ValueError('Các thuộc tính đều không biến thiên; PCA không có ý nghĩa.')
        pca = PCA(n_components=2).fit(scaled)
        X_plot = pca.transform(scaled)
        X_model = X_plot.copy()
        axes = ('PCA 1', 'PCA 2')
        notes.append(f'PCA giữ {pca.explained_variance_ratio_.sum():.1%} phương sai của 4 thuộc tính đã chuẩn hóa.')
        preprocessing = 'Chuẩn hóa 4 thuộc tính → PCA 2D'
    else:
        X_plot = sample[list(columns)].to_numpy(float)
        axes = tuple(AXES[c] for c in columns)
        X_model = X_plot.copy()
        if standardize:
            transformer = StandardScaler().fit(X_model)
            X_model = transformer.transform(X_model)
        preprocessing = 'StandardScaler trên 2 thuộc tính' if standardize else 'Giữ thang đo gốc'
    return dict(X=X_model, plot=X_plot, axes=axes, frame=sample, truth=None,
                scaler=transformer, notes=notes, preprocessing=preprocessing, total=len(df))


def prepare_dataset(name, n=300, seed=42, pair='Tuổi vs Điểm chi tiêu',
                    standardize=True, source=None):
    if name == DATASETS[0]:
        return customer_data(source, pair, n, seed, standardize)
    X_plot, truth, suggested_k = synthetic_data(name, n, seed)
    scaler = StandardScaler().fit(X_plot) if standardize else None
    X = scaler.transform(X_plot) if scaler else X_plot.copy()
    return dict(X=X, plot=X_plot, axes=('Tọa độ X', 'Tọa độ Y'), frame=None, truth=truth,
                scaler=scaler, notes=[], total=len(X), suggested_k=suggested_k,
                preprocessing='StandardScaler trên 2 thuộc tính' if scaler else 'Giữ thang đo gốc')
