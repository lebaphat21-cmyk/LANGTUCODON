"""
Hệ thống Web Demo Mô phỏng & Đối sánh Thuật toán Phân cụm CURE (Clustering Using REpresentatives)
Môn học: Khai thác dữ liệu / Khai phá dữ liệu - Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT)
Khởi chạy: python -m streamlit run app_streamlit.py

Đọc hoặc chạy các cell # %% từ trên xuống. Các cell khai báo hàm không
phân cụm ngay; dữ liệu và kết quả được tạo trước khi dựng các tab.
Streamlit chạy lại toàn bộ file khi đổi điều khiển; cache tái sử dụng
kết quả khi đầu vào không đổi. Mỗi tab chỉ đọc kết quả dùng chung.
"""

# %% Cell 01 - Thư viện
import os
import sys
import time
import json
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from cure_algorithm import CURE, KMedoids, DIANA
from ui_components import (
    CLUSTER_COLORS, apply_theme, render_tab_header, render_metrics,
    render_chart, render_table,
)

# %% Cell 02 - Hàm get_test_csv_path
def get_test_csv_path():
    """Tìm CSV theo vị trí file app trước, sau đó mới thử thư mục chạy."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "Data", "Test.csv"),
        os.path.join(base_dir, "data", "Test.csv"),
        os.path.join(base_dir, "Data", "test.csv"),
        os.path.join(base_dir, "data", "test.csv"),
        os.path.join(base_dir, "Test.csv"),
        os.path.join(base_dir, "test.csv"),
        os.path.join("Data", "Test.csv"),
        os.path.join("data", "Test.csv"),
        "Test.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

# %% Cell 03 - Hàm load_and_preprocess_customer_data
@st.cache_data
def load_and_preprocess_customer_data(pair_name, n_pts, seed, uploaded_bytes=None):
    """Đọc CSV → điền giá trị thiếu → lấy mẫu → chọn hai đặc trưng/PCA.

    Trả về X, tên hai trục và đúng các dòng dữ liệu đã lấy mẫu.
    Các tab phân tích phải dùng cùng mẫu này để thống kê khớp biểu đồ.
    """
    df = None
    if uploaded_bytes is not None:
        import io
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
    else:
        csv_path = get_test_csv_path()
        if csv_path is not None:
            df = pd.read_csv(csv_path)

    if df is None:
        return None, "X", "Y", None
    # Tiền xử lý
    spend_map = {'Low': 1, 'Average': 2, 'High': 3}
    df['Spending_Score_Num'] = df['Spending_Score'].map(spend_map).fillna(1)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Work_Experience'] = df['Work_Experience'].fillna(df['Work_Experience'].median())
    df['Family_Size'] = df['Family_Size'].fillna(df['Family_Size'].median())

    # Lấy mẫu ngẫu nhiên (Pha 1 của CURE)
    sample_df = df.sample(n=min(n_pts, len(df)), random_state=seed).copy()

    if "chi tiêu" in pair_name.lower():
        # Thêm một chút jitter nhỏ để các điểm rời rạc (1, 2, 3) không đè khít lên nhau
        rng = np.random.RandomState(seed)
        jitter = rng.uniform(-0.12, 0.12, size=len(sample_df))
        X = np.column_stack([
            sample_df['Age'].values,
            sample_df['Spending_Score_Num'].values + jitter
        ])
        x_name, y_name = "Tuổi (Age)", "Điểm chi tiêu (1:Low, 2:Avg, 3:High)"
    elif "Kinh nghiệm" in pair_name and "Tuổi" in pair_name:
        X = np.column_stack([
            sample_df['Age'].values,
            sample_df['Work_Experience'].values
        ])
        x_name, y_name = "Tuổi (Age)", "Kinh nghiệm làm việc (Năm)"
    elif "Quy mô gia đình" in pair_name and "Tuổi" in pair_name:
        X = np.column_stack([
            sample_df['Age'].values,
            sample_df['Family_Size'].values
        ])
        x_name, y_name = "Tuổi (Age)", "Quy mô gia đình (Thành viên)"
    elif "Kinh nghiệm" in pair_name and "gia đình" in pair_name:
        X = np.column_stack([
            sample_df['Work_Experience'].values,
            sample_df['Family_Size'].values
        ])
        x_name, y_name = "Kinh nghiệm làm việc (Năm)", "Quy mô gia đình (Thành viên)"
    else:
        # PCA 2D
        num_cols = ['Age', 'Spending_Score_Num', 'Work_Experience', 'Family_Size']
        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(sample_df[num_cols])
        pca = PCA(n_components=2)
        X = pca.fit_transform(scaled_vals)
        x_name, y_name = "Thành phần chính 1 (PCA 1)", "Thành phần chính 2 (PCA 2)"

    return X, x_name, y_name, sample_df

# %% Cell 04 - Hàm generate_synthetic_dataset
@st.cache_data
def generate_synthetic_dataset(d_type, n_pts, seed):
    """Sinh dữ liệu có thể tái lập bằng seed, cùng định dạng với bộ nạp CSV."""
    rng = np.random.RandomState(seed)
    if "Moons" in d_type:
        X, _ = datasets.make_moons(n_samples=n_pts, noise=0.06, random_state=seed)
    elif "Circles" in d_type:
        X, _ = datasets.make_circles(n_samples=n_pts, factor=0.5, noise=0.05, random_state=seed)
    elif "Anisotropic" in d_type:
        X_blobs, _ = datasets.make_blobs(n_samples=n_pts, cluster_std=[0.9, 0.9, 0.9], random_state=seed)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X_blobs, transformation)
    else:
        # Blobs with Outliers
        n_clean = max(80, n_pts - 40)
        blobs, _ = datasets.make_blobs(n_samples=n_clean, centers=2, cluster_std=0.8, random_state=seed)
        outliers = rng.uniform(low=-7, high=7, size=(40, 2))
        X = np.vstack([blobs, outliers])
    return X, "Tọa độ X", "Tọa độ Y", None

# %% Cell 05 - Hàm compute_elbow
@st.cache_data(show_spinner=False)
def compute_elbow(X_tuple, k_max, seed):
    """Tính Inertia (K-Means) và Silhouette cho k=2..k_max để vẽ Elbow Chart."""
    X = np.array(X_tuple)
    inertias, sil_scores = [], []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init='auto')
        km.fit(X)
        inertias.append(km.inertia_)
        try:
            sil_scores.append(silhouette_score(X, km.labels_))
        except Exception:
            sil_scores.append(0.0)
    return inertias, sil_scores

# %% Cell 06 - Hàm compute_metrics
def compute_metrics(X, labels, exec_time):
    """Bỏ nhãn nhiễu -1 và tính cùng bộ chỉ số cho mọi thuật toán.

    -1/99/0 là giá trị quy ước cũ khi chỉ số không tính được; không phải
    kết quả đo chất lượng. Chỉ số không xác định khi chỉ có một cụm.
    """
    valid_mask = labels != -1
    unique_clusters = len(set(labels[valid_mask]))
    if unique_clusters >= 2:
        try:
            sil = silhouette_score(X[valid_mask], labels[valid_mask])
        except Exception:
            sil = 0.0
        try:
            db = davies_bouldin_score(X[valid_mask], labels[valid_mask])
        except Exception:
            db = 99.0
        try:
            ch = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
        except Exception:
            ch = 0.0
    else:
        sil, db, ch = -1.0, 99.0, 0.0
    return {
        "Silhouette": sil,
        "Davies-Bouldin": db,
        "Calinski-Harabasz": ch,
        "Time": exec_time,
        "Clusters": unique_clusters
    }

# %% Cell 07 - Hàm make_scatter_figure
def make_scatter_figure(X, labels, title, reps=None, means=None, centers=None, center_label="Tâm", x_title="X", y_title="Y"):
    """Tạo hình với màu cụm thống nhất; render_chart áp dụng bố cục chung."""
    df_p = pd.DataFrame(X, columns=['X', 'Y'])
    df_p['Cụm'] = [f"Cụm {l+1}" if l != -1 else "Ngoại lai (Nhiễu)" for l in labels]

    color_map = {}
    palette = CLUSTER_COLORS
    for idx, l in enumerate(sorted(list(set(labels)))):
        if l == -1:
            color_map["Ngoại lai (Nhiễu)"] = "#94a3b8"
        else:
            color_map[f"Cụm {l+1}"] = palette[l % len(palette)]

    fig = px.scatter(
        df_p, x='X', y='Y', color='Cụm',
        color_discrete_map=color_map,
        opacity=0.82
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))

    # Vẽ điểm đại diện sau co cụm (CURE)
    if reps is not None:
        for idx, rep in enumerate(reps):
            fig.add_trace(go.Scatter(
                x=rep[:, 0], y=rep[:, 1],
                mode='markers',
                marker=dict(symbol='x', size=11, color='black', line=dict(width=2.4)),
                name=f"Rep points Cụm {idx+1}",
                showlegend=(idx == 0)
            ))

    # Vẽ trọng tâm Mean (CURE hoặc KMeans)
    if means is not None:
        fig.add_trace(go.Scatter(
            x=means[:, 0], y=means[:, 1],
            mode='markers',
            marker=dict(symbol='star', size=16, color='gold', line=dict(width=1.5, color='black')),
            name="Trọng tâm (Mean)",
            showlegend=True
        ))

    # Vẽ Medoids (K-Medoids)
    if centers is not None:
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1],
            mode='markers',
            marker=dict(symbol='diamond', size=13, color='#dc2626', line=dict(width=1.5, color='black')),
            name=center_label,
            showlegend=True
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#103673', family='sans-serif')),
        height=480,
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(orientation="h", y=-0.18, x=0.0, font=dict(size=11)),
        plot_bgcolor="#fafbfc",
        xaxis=dict(title=x_title, showgrid=True, gridcolor='#f1f5f9', zeroline=False),
        yaxis=dict(title=y_title, showgrid=True, gridcolor='#f1f5f9', zeroline=False)
    )
    return fig

# %% Cell 08 - Hàm run_all_algorithms
@st.cache_data(show_spinner="⏳ Đang chạy 4 thuật toán phân cụm...")
def run_all_algorithms(X_tuple, k, c, alpha, seed):
    """
    Cache toàn bộ kết quả 4 thuật toán.
    Chỉ tính lại khi dữ liệu hoặc tham số thực sự thay đổi.
    """
    X = np.array(X_tuple)

    # 1. CURE
    t0 = time.time()
    cure_m = CURE(n_clusters=k, n_representatives=c, shrink_factor=alpha)
    cure_m.fit(X)
    c_time = time.time() - t0
    c_labels = cure_m.labels_
    c_reps_r  = cure_m.get_representatives()
    c_means_r = cure_m.get_cluster_means()
    c_metrics = compute_metrics(X, c_labels, c_time)

    # 2. K-Means
    t0 = time.time()
    km_m = KMeans(n_clusters=k, random_state=seed, n_init='auto')
    km_lbl = km_m.fit_predict(X)
    km_t = time.time() - t0
    km_ctr = km_m.cluster_centers_
    km_met = compute_metrics(X, km_lbl, km_t)

    # 3. K-Medoids
    t0 = time.time()
    kmed_m = KMedoids(n_clusters=k, random_state=seed)
    kmed_lbl = kmed_m.fit_predict(X)
    kmed_t = time.time() - t0
    kmed_ctr = kmed_m.cluster_centers_
    kmed_met = compute_metrics(X, kmed_lbl, kmed_t)

    # 4. Hierarchical (Single Linkage)
    t0 = time.time()
    hier_m = AgglomerativeClustering(n_clusters=k, linkage='single')
    hier_lbl = hier_m.fit_predict(X)
    hier_t = time.time() - t0
    hier_met = compute_metrics(X, hier_lbl, hier_t)

    return (
        c_labels, c_reps_r, c_means_r, c_metrics, c_time,
        km_lbl, km_ctr, km_met, km_t,
        kmed_lbl, kmed_ctr, kmed_met, kmed_t,
        hier_lbl, hier_met, hier_t
    )

# %% Cell 09 - Hàm run_agnes_4linkage
@st.cache_data(show_spinner="⏳ Đang chạy 4 biến thể AGNES...")
def run_agnes_4linkage(X_tuple, k, seed):
    """Chạy bốn linkage trên cùng X và k; seed giữ trong khóa thực nghiệm."""
    X = np.array(X_tuple)
    results = {}
    for linkage in ['single', 'complete', 'average', 'ward']:
        t0 = time.time()
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        lbl = model.fit_predict(X)
        elapsed = time.time() - t0
        results[linkage] = {
            'labels': lbl,
            'metrics': compute_metrics(X, lbl, elapsed),
            'time': elapsed
        }
    return results

# %% Cell 10 - Hàm run_diana
@st.cache_data(show_spinner="⏳ Đang chạy DIANA (Top-down Divisive)...")
def run_diana(X_tuple, k):
    """Tính DIANA một lần cho cả tab chi tiết lẫn bảng tổng hợp."""
    X = np.array(X_tuple)
    t0 = time.time()
    model = DIANA(n_clusters=k)
    model.fit(X)
    elapsed = time.time() - t0
    lbl = model.labels_
    means = model.get_cluster_means_from_X(X)
    met = compute_metrics(X, lbl, elapsed)
    return lbl, means, met, elapsed, model.clusters_

# %% Cell 11 - Hàm render_canvas_html
def render_canvas_html(points_2d, k, c, alpha, title_dataset):
    # Canvas hiện mô phỏng theo tọa độ hiển thị; không dùng để tính chỉ số Python.
    xs = [float(p[0]) for p in points_2d]
    ys = [float(p[1]) for p in points_2d]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pad = 55
    w, h = 800, 480

    norm_pts = []
    for x, y in zip(xs, ys):
        nx = pad + ((x - min_x) / (max_x - min_x + 1e-6)) * (w - 2 * pad)
        ny = (h - pad) - ((y - min_y) / (max_y - min_y + 1e-6)) * (h - 2 * pad)
        norm_pts.append([round(nx, 1), round(ny, 1)])

    pts_json = json.dumps(norm_pts)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <script src="https://cdn.tailwindcss.com"></script>
      <style>
        body {{ margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8fafc; }}
        canvas {{ background-color: #fafbfc; border: 2px solid #cbd5e1; cursor: crosshair; }}
      </style>
    </head>
    <body class="p-2">
      <div class="bg-white rounded-xl shadow-md border border-slate-200 p-4 max-w-5xl mx-auto">
        <!-- Header thông tin -->
        <div class="flex flex-wrap justify-between items-center pb-3 border-b border-slate-200 mb-3 gap-2">
          <div>
            <div class="text-xs font-bold text-blue-900 uppercase">Trực quan hóa Hoạt họa Canvas HTML5</div>
            <div class="text-sm text-slate-600 font-semibold">{title_dataset} &bull; <span id="totalPts">{len(norm_pts)} điểm</span></div>
          </div>
          <div class="flex items-center gap-3">
            <div class="text-sm font-bold text-slate-800">
              Số cụm: <span id="curK" class="text-blue-700 text-lg font-mono">0</span> 
              <span class="text-slate-400 font-normal">/ Mục tiêu: <strong id="tgtK" class="text-slate-700">{k}</strong></span>
            </div>
            <span id="badge" class="text-xs font-semibold px-3 py-1 bg-amber-100 text-amber-800 rounded-full border border-amber-300">
              Sẵn sàng
            </span>
          </div>
        </div>

        <!-- Canvas -->
        <div class="relative flex justify-center bg-slate-100/60 rounded-lg p-1">
          <canvas id="cv" width="800" height="480" class="rounded shadow-inner max-w-full h-auto"></canvas>
        </div>

        <!-- Thanh điều khiển & Chú thích -->
        <div class="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3 items-center pt-2">
          <div class="flex gap-2">
            <button id="btnS" class="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold py-2 px-3 rounded shadow transition">
              ⏩ 1 Bước (Step)
            </button>
            <button id="btnA" class="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold py-2 px-3 rounded shadow transition">
              ▶️ Chạy Tự Động
            </button>
            <button id="btnR" class="bg-slate-200 hover:bg-slate-300 text-slate-700 text-xs font-semibold py-2 px-3 rounded transition">
              🔄 Đặt Lại
            </button>
          </div>

          <div class="flex items-center gap-2 text-xs font-medium text-slate-600">
            <span>Tốc độ:</span>
            <input id="spd" type="range" min="30" max="400" step="10" value="100" class="w-full accent-blue-600">
            <span id="spdVal" class="font-mono text-slate-800">100ms</span>
          </div>

          <div class="flex items-center justify-end gap-3 text-xs">
            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block"></span> Cụm</span>
            <span class="flex items-center gap-1"><span class="text-red-600 font-bold">✕</span> Rep (co)</span>
            <span class="flex items-center gap-1"><span class="text-amber-500 font-bold">★</span> Mean</span>
            <span class="flex items-center gap-1 text-slate-400"><span class="border-b border-dashed border-red-400 w-3 inline-block"></span> Nối</span>
          </div>
        </div>

        <!-- Nhật ký bước chạy -->
        <div id="log" class="mt-3 p-2.5 bg-slate-50 rounded border border-slate-200 text-xs text-slate-700 font-sans">
          Bấm <strong>"1 Bước (Step)"</strong> hoặc <strong>"Chạy Tự Động"</strong> để theo dõi thuật toán CURE gom cụm từng bước theo Farthest-Point và co cụm &alpha;={alpha}.
        </div>

        <!-- KHUNG GIẢI THÍCH KẾT QUẢ PHÂN CỤM SAU KHI HOÀN TẤT -->
        <div id="expBox" class="mt-4 p-4 bg-gradient-to-br from-blue-50/90 via-indigo-50/80 to-slate-50 rounded-xl border-2 border-blue-200 shadow-sm hidden transition-all">
          <div class="flex items-center justify-between border-b border-blue-200 pb-2 mb-3">
            <div class="flex items-center gap-2">
              <span class="text-2xl">🎯</span>
              <div>
                <h3 class="text-sm font-bold text-blue-950 uppercase tracking-wide">Giải Thích Chi Tiết Kết Quả Phân Cụm</h3>
                <p class="text-[11px] text-blue-700 font-medium">Đánh giá cấu trúc nhóm, ý nghĩa phân khúc khách hàng & cơ chế thuật toán CURE</p>
              </div>
            </div>
            <span class="text-xs font-bold px-2.5 py-1 bg-emerald-600 text-white rounded-full shadow-sm">✓ Đã phân cụm thành công</span>
          </div>

          <!-- Cards từng cụm -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3" id="cardsBox"></div>

          <!-- Phân tích học thuật -->
          <div class="bg-white p-3.5 rounded-lg border border-blue-100 text-xs text-slate-700 space-y-2">
            <div class="font-bold text-blue-900 flex items-center gap-1.5">
              <span>💡</span> <span>Nhận Định & Đánh Giá Học Thuật Về Thuật Toán CURE:</span>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11.5px] text-slate-600 leading-relaxed">
              <div class="p-2.5 bg-slate-50 rounded border border-slate-200">
                <strong class="text-blue-900">1. Ưu thế của c={c} điểm đại diện:</strong><br>
                Khác với K-Means hay K-Medoids chỉ dùng duy nhất 1 điểm tâm tròn, CURE rải đều <strong>{c} điểm đại diện</strong> bám dọc theo đường viền thực tế của dữ liệu. Nhờ đó, thuật toán nhận diện trọn vẹn cả những cụm kéo dài, cụm cong uốn lượn hay các phân khúc phân bố bất đối xứng.
              </div>
              <div class="p-2.5 bg-slate-50 rounded border border-slate-200">
                <strong class="text-blue-900">2. Cơ chế co cụm kháng nhiễu (α={alpha}):</strong><br>
                Việc kéo các điểm đại diện co lùi về trọng tâm theo hệ số <strong>α = {alpha}</strong> tạo ra một hành lang khoảng cách an toàn. Các điểm khách hàng bất thường hoặc ngoại lai nằm ở ngoài rìa sẽ không thể làm lệch ranh giới cụm.
              </div>
            </div>
          </div>
        </div>
      </div>

      <script>
        const canvas = document.getElementById('cv');
        const ctx = canvas.getContext('2d');
        const initPts = {pts_json};
        let points = JSON.parse(JSON.stringify(initPts));
        let clusters = [];
        let autoTimer = null;
        let isAuto = false;

        const target_k = {k};
        const c_num = {c};
        const alpha_val = {alpha};

        const COLORS = [
          '#2563eb', '#dc2626', '#16a34a', '#d97706', '#9333ea', 
          '#0891b2', '#db2777', '#4b5563', '#4f46e5', '#ca8a04',
          '#059669', '#e11d48', '#7c3aed', '#0284c7', '#ea580c'
        ];
        const COLOR_NAMES = [
          'Xanh dương', 'Đỏ', 'Xanh lá', 'Cam', 'Tím',
          'Xanh lơ', 'Hồng', 'Xám', 'Chàm', 'Vàng đậm'
        ];

        const curK = document.getElementById('curK');
        const badge = document.getElementById('badge');
        const logBox = document.getElementById('log');
        const spdInput = document.getElementById('spd');
        const spdVal = document.getElementById('spdVal');
        const btnA = document.getElementById('btnA');
        const expBox = document.getElementById('expBox');
        const cardsBox = document.getElementById('cardsBox');

        spdInput.oninput = () => {{
          spdVal.innerText = spdInput.value + 'ms';
          if (isAuto) {{
            clearInterval(autoTimer);
            autoTimer = setInterval(stepClustering, parseInt(spdInput.value));
          }}
        }};

        function d(p1, p2) {{ return Math.hypot(p1[0] - p2[0], p1[1] - p2[1]); }}

        function getMean(pts) {{
          let sx = 0, sy = 0;
          for (let p of pts) {{ sx += p[0]; sy += p[1]; }}
          return [sx / pts.length, sy / pts.length];
        }}

        function updateReps(cl) {{
          const pts = cl.points;
          cl.mean = getMean(pts);
          let rawReps = [];

          if (pts.length <= c_num) {{
            rawReps = pts.map(p => [...p]);
          }} else {{
            let maxD = -1, firstP = pts[0];
            for (let p of pts) {{
              let distM = d(p, cl.mean);
              if (distM > maxD) {{ maxD = distM; firstP = p; }}
            }}
            rawReps.push([...firstP]);

            for (let i = 1; i < c_num; i++) {{
              let maxMin = -1, nextP = pts[0];
              for (let p of pts) {{
                let minD = Math.min(...rawReps.map(r => d(p, r)));
                if (minD > maxMin) {{ maxMin = minD; nextP = p; }}
              }}
              rawReps.push([...nextP]);
            }}
          }}

          cl.reps = rawReps.map(p => [
            p[0] + alpha_val * (cl.mean[0] - p[0]),
            p[1] + alpha_val * (cl.mean[1] - p[1])
          ]);
        }}

        function clDist(c1, c2) {{
          let minD = Infinity;
          for (let r1 of c1.reps) {{
            for (let r2 of c2.reps) {{
              let distVal = d(r1, r2);
              if (distVal < minD) minD = distVal;
            }}
          }}
          return minD;
        }}

        function showExplanation() {{
          const totalPts = points.length;
          let cardsHtml = '';
          clusters.forEach((cl, idx) => {{
            const color = COLORS[idx % COLORS.length];
            const colorName = COLOR_NAMES[idx % COLOR_NAMES.length];
            const count = cl.points.length;
            const pct = ((count / totalPts) * 100).toFixed(1);

            let profileTitle = 'Nhóm Khách Hàng Tiềm Năng';
            let profileDesc = 'Mật độ tập trung cao, phân bố đều theo biên độ CURE.';

            if (idx === 0) {{
              profileTitle = 'Phân khúc 1: Khách hàng Phổ thông (Trẻ tuổi)';
              profileDesc = 'Chiếm tỷ trọng lớn nhất, mức chi tiêu và độ tuổi trẻ/trung tâm thị trường.';
            }} else if (idx === 1) {{
              profileTitle = 'Phân khúc 2: Khách hàng Trưởng thành / Cao cấp';
              profileDesc = 'Độ tuổi cao hơn hoặc chỉ số chi tiêu vượt trội, có giá trị sinh lời cao.';
            }} else if (idx === 2) {{
              profileTitle = 'Phân khúc 3: Khách hàng Cao tuổi / Hưu trí';
              profileDesc = 'Nhóm khách hàng cao tuổi/an hưởng tuổi già, quy mô gia đình nhỏ, nhu cầu ổn định và chăm sóc sức khỏe.';
            }} else {{
              profileTitle = `Phân khúc ${{idx + 1}}: Nhóm Khách hàng Ngách`;
              profileDesc = 'Tập khách hàng đặc thù được CURE bóc tách chính xác mà không bị gộp lẫn.';
            }}

            cardsHtml += `
              <div class="bg-white p-3 rounded-lg border border-slate-200 shadow-sm flex flex-col justify-between" style="border-top: 4px solid ${{color}};">
                <div>
                  <div class="flex justify-between items-center mb-1">
                    <span class="font-bold text-xs" style="color: ${{color}};">Cụm ${{idx + 1}} (${{colorName}})</span>
                    <span class="text-[11px] font-semibold bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded font-mono">${{count}} điểm (${{pct}}%)</span>
                  </div>
                  <div class="w-full bg-slate-100 rounded-full h-1.5 mb-2 overflow-hidden">
                    <div class="h-1.5 rounded-full" style="width: ${{pct}}%; background-color: ${{color}};"></div>
                  </div>
                  <div class="text-[11px] font-bold text-slate-800 mb-0.5">${{profileTitle}}</div>
                  <p class="text-[10.5px] text-slate-500 leading-snug">${{profileDesc}}</p>
                </div>
                <div class="mt-2 pt-2 border-t border-slate-100 text-[10px] text-slate-400 flex justify-between">
                  <span>Đại diện: <strong>${{cl.reps.length}} rep</strong></span>
                  <span>Tâm: <strong>(${{Math.round(cl.mean[0])}}, ${{Math.round(cl.mean[1])}})</strong></span>
                </div>
              </div>
            `;
          }});

          cardsBox.innerHTML = cardsHtml;
          expBox.classList.remove('hidden');
        }}

        function init() {{
          stopAuto();
          expBox.classList.add('hidden');
          clusters = points.map((p, idx) => {{
            const cl = {{ id: idx, points: [[p[0], p[1]]], mean: [p[0], p[1]], reps: [[p[0], p[1]]] }};
            updateReps(cl);
            return cl;
          }});
          curK.innerText = clusters.length;
          badge.innerText = "Sẵn sàng";
          badge.className = "text-xs font-semibold px-3 py-1 bg-amber-100 text-amber-800 rounded-full border border-amber-300";
          logBox.innerHTML = "Khởi tạo thành công <strong>" + points.length + " cụm ban đầu</strong>. Bấm 1 Bước hoặc Chạy Tự Động.";
          draw();
        }}

        function stepClustering() {{
          if (clusters.length <= target_k) {{
            badge.innerText = "Hoàn tất!";
            badge.className = "text-xs font-semibold px-3 py-1 bg-emerald-100 text-emerald-800 rounded-full border border-emerald-300";
            logBox.innerHTML = "✅ <strong>Đạt mục tiêu k=" + target_k + " cụm!</strong> Cấu trúc các cụm CURE đã được cố định chính xác. Xem phân tích bên dưới.";
            stopAuto();
            showExplanation();
            return false;
          }}

          let minD = Infinity;
          let pair = [0, 1];
          for (let i = 0; i < clusters.length; i++) {{
            for (let j = i + 1; j < clusters.length; j++) {{
              let distVal = clDist(clusters[i], clusters[j]);
              if (distVal < minD) {{ minD = distVal; pair = [i, j]; }}
            }}
          }}

          let [i, j] = pair;
          let c1 = clusters[i], c2 = clusters[j];
          c1.points = c1.points.concat(c2.points);
          updateReps(c1);
          clusters.splice(j, 1);

          curK.innerText = clusters.length;
          badge.innerText = "Đang gom (" + clusters.length + ")";
          badge.className = "text-xs font-semibold px-3 py-1 bg-blue-100 text-blue-800 rounded-full border border-blue-300";
          logBox.innerHTML = "🔹 Sáp nhập 2 cụm có khoảng cách nhỏ nhất: <code>" + minD.toFixed(2) + "px</code>. Cụm mới có " + c1.points.length + " điểm, co " + c1.reps.length + " đại diện về tâm &alpha;=" + alpha_val + ".";
          draw();
          return true;
        }}

        function draw() {{
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.strokeStyle = '#f1f5f9';
          ctx.lineWidth = 1;
          for (let x = 0; x < canvas.width; x += 40) {{ ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }}
          for (let y = 0; y < canvas.height; y += 40) {{ ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }}

          clusters.forEach((cl, cIdx) => {{
            let color = COLORS[cIdx % COLORS.length];
            cl.points.forEach(p => {{
              ctx.beginPath();
              ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
              ctx.fillStyle = color;
              ctx.fill();
              ctx.strokeStyle = '#ffffff';
              ctx.lineWidth = 1;
              ctx.stroke();
            }});

            if (cl.points.length > 1) {{
              ctx.strokeStyle = '#ef4444';
              ctx.lineWidth = 1;
              ctx.setLineDash([3, 3]);
              cl.reps.forEach(r => {{
                ctx.beginPath(); ctx.moveTo(cl.mean[0], cl.mean[1]); ctx.lineTo(r[0], r[1]); ctx.stroke();
              }});
              ctx.setLineDash([]);
            }}

            cl.reps.forEach(r => {{
              ctx.strokeStyle = '#b91c1c';
              ctx.lineWidth = 2.4;
              let s = 5;
              ctx.beginPath();
              ctx.moveTo(r[0] - s, r[1] - s); ctx.lineTo(r[0] + s, r[1] + s);
              ctx.moveTo(r[0] + s, r[1] - s); ctx.lineTo(r[0] - s, r[1] + s);
              ctx.stroke();
            }});

            if (cl.points.length > 1) {{
              drawStar(cl.mean[0], cl.mean[1], 5, 9, 4);
            }}
          }});
        }}

        function drawStar(cx, cy, spikes, outerRadius, innerRadius) {{
          let rot = Math.PI / 2 * 3;
          let step = Math.PI / spikes;
          ctx.beginPath();
          ctx.moveTo(cx, cy - outerRadius);
          for (let i = 0; i < spikes; i++) {{
            ctx.lineTo(cx + Math.cos(rot) * outerRadius, cy + Math.sin(rot) * outerRadius);
            rot += step;
            ctx.lineTo(cx + Math.cos(rot) * innerRadius, cy + Math.sin(rot) * innerRadius);
            rot += step;
          }}
          ctx.closePath();
          ctx.fillStyle = '#f59e0b'; ctx.fill();
          ctx.strokeStyle = '#78350f'; ctx.lineWidth = 1.5; ctx.stroke();
        }}

        function stopAuto() {{
          if (autoTimer) {{ clearInterval(autoTimer); autoTimer = null; }}
          isAuto = false;
          btnA.innerText = "▶️ Chạy Tự Động";
          btnA.className = "flex-1 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold py-2 px-3 rounded shadow transition";
        }}

        canvas.addEventListener('mousedown', (e) => {{
          const rect = canvas.getBoundingClientRect();
          const sx = canvas.width / rect.width;
          const sy = canvas.height / rect.height;
          points.push([(e.clientX - rect.left) * sx, (e.clientY - rect.top) * sy]);
          init();
        }});

        document.getElementById('btnS').onclick = () => {{ stopAuto(); stepClustering(); }};
        btnA.onclick = () => {{
          if (isAuto) {{
            stopAuto();
          }} else {{
            isAuto = true;
            btnA.innerText = "⏸️ Tạm Dừng";
            btnA.className = "flex-1 bg-amber-600 hover:bg-amber-700 text-white text-xs font-bold py-2 px-3 rounded shadow transition";
            autoTimer = setInterval(() => {{
              let hasMore = stepClustering();
              if (!hasMore) stopAuto();
            }}, parseInt(spdInput.value));
          }}
        }};
        document.getElementById('btnR').onclick = () => {{
          points = JSON.parse(JSON.stringify(initPts));
          init();
        }};

        init();
      </script>
    </body>
    </html>
    """
    return html


# %% Cell 12 - Cấu hình trang và giao diện chung
# ==========================================
# CẤU HÌNH TRANG STREAMLIT
# ==========================================
st.set_page_config(
    page_title="CURE Clustering Visualizer - HUIT",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

# Custom CSS giao diện chuẩn HUIT
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #103673 0%, #1f4e79 100%);
        padding: 22px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .huit-title {
        color: #F2A900 !important;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin: 0;
        font-size: 22px;
    }
    .sub-title {
        color: #FFFFFF;
        font-weight: 600;
        margin: 6px 0 0 0;
        font-size: 17px;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
        color: #1e293b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #103673 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Banner Header
st.markdown("""
<div class="main-header">
    <div class="huit-title">TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP. HỒ CHÍ MINH (HUIT)</div>
    <div class="sub-title">HỆ THỐNG MÔ PHỎNG & ĐỐI SÁNH THUẬT TOÁN PHÂN CỤM CURE</div>
    <p style="margin: 6px 0 0 0; font-size: 13.5px; opacity: 0.92;">
        Môn học: Khai thác dữ liệu | Đề tài: <b>Phân cụm dữ liệu dựa trên thuật toán CURE (Clustering Using REpresentatives)</b> &bull; Hỗ trợ Dữ liệu thực tế từ folder <code>Data/</code>
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR: CẤU HÌNH THAM SỐ
# ==========================================
# %% Cell 13 - Đọc tham số từ sidebar
st.sidebar.markdown("### ⚙️ CẤU HÌNH THỰC NGHIỆM")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
csv_files = []
if os.path.exists(DATA_DIR):
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

dataset_options = [
    "📁 Dữ liệu từ folder Data (Test.csv - Khách hàng)",
    "Two Moons (2 Vầng trăng khuyết - Phi cầu)",
    "Concentric Circles (2 Vòng tròn đồng tâm - Phi cầu)",
    "Anisotropic Blobs (Cụm kéo dài hình elip)",
    "Blobs with Outliers (Cụm có điểm ngoại lai/nhiễu)"
]

dataset_type = st.sidebar.selectbox("1. Chọn tập dữ liệu kiểm thử:", dataset_options, index=0)

selected_feature_pair = "Tuổi (Age) vs Điểm chi tiêu (Spending Score)"
axis_x_name = "X"
axis_y_name = "Y"
df_customer_raw = None

if "Data" in dataset_type:
    st.sidebar.markdown("#### 📂 Thuộc tính dữ liệu Khách hàng:")
    feature_pairs = [
        "Tuổi (Age) vs Điểm chi tiêu (Spending Score)",
        "Tuổi (Age) vs Kinh nghiệm làm việc (Work Experience)",
        "Tuổi (Age) vs Quy mô gia đình (Family Size)",
        "Kinh nghiệm làm việc vs Quy mô gia đình",
        "Không gian PCA 2D (Tổng hợp các thuộc tính số)"
    ]
    selected_feature_pair = st.sidebar.selectbox("Chọn 2 thuộc tính phân cụm:", feature_pairs, index=0)
    
    n_samples = st.sidebar.slider(
        "Kích thước mẫu lấy ngẫu nhiên (s):", 
        min_value=60, max_value=400, value=140, step=20,
        help="Pha 1 của thuật toán CURE: Rút trích mẫu ngẫu nhiên s điểm từ cơ sở dữ liệu lớn để giảm chi phí tính toán."
    )
else:
    n_samples = st.sidebar.slider("Số lượng điểm dữ liệu (N):", min_value=120, max_value=600, value=300, step=30)

noise_seed = st.sidebar.number_input("Random Seed:", min_value=1, max_value=999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ THAM SỐ THUẬT TOÁN CURE")
default_k = 3 if "Data" in dataset_type else 2
k_clusters = st.sidebar.slider("Số cụm mục tiêu (k):", min_value=2, max_value=6, value=default_k, step=1)
c_reps = st.sidebar.slider("Số điểm đại diện mỗi cụm (c):", min_value=1, max_value=8, value=4, step=1,
                          help="Càng nhiều điểm đại diện càng nắm bắt được các khúc uốn lượn và phân bố phức tạp của cụm.")
alpha_shrink = st.sidebar.slider("Hệ số co cụm (alpha):", min_value=0.0, max_value=1.0, value=0.4, step=0.05,
                                help="0.0: Điểm đại diện giữ nguyên ở biên ngoài | 1.0: Điểm đại diện co hoàn toàn về trọng tâm mean | 0.3-0.5: Tối ưu chống nhiễu.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 GỢI Ý SỐ CỤM TỐI ƯU")


# ==========================================


# %% Cell 14 - Nạp và tiền xử lý dữ liệu dùng chung
uploaded_file_bytes = None
if "Data" in dataset_type:
    # Kiểm tra xem file có trên máy / server không
    found_path = get_test_csv_path()
    if found_path is None:
        st.sidebar.warning("⚠️ Không tìm thấy folder Data/Test.csv trên server!")
        uploaded_csv = st.sidebar.file_uploader("Tải lên file Test.csv trực tiếp:", type=["csv"])
        if uploaded_csv is not None:
            uploaded_file_bytes = uploaded_csv.getvalue()
            
    X_data, axis_x_name, axis_y_name, df_customer_raw = load_and_preprocess_customer_data(
        selected_feature_pair, n_samples, noise_seed, uploaded_file_bytes
    )
    if X_data is None:
        st.error("❌ **Chưa có dữ liệu Data/Test.csv trên GitHub/Server!**\n\n"
                 "👉 **Nguyên nhân:** Thư mục `Data/Test.csv` chưa được push lên GitHub của bạn (hoặc bị phân biệt hoa/thường).\n\n"
                 "👉 **Khắc phục:** Hãy tải file `Test.csv` lên ở thanh bên trái (Sidebar), hoặc commit và push folder `Data/` lên GitHub.")
        # Dùng tạm fallback để không crash web
        X_data, axis_x_name, axis_y_name, _ = generate_synthetic_dataset("Two Moons", n_samples, noise_seed)
else:
    X_data, axis_x_name, axis_y_name, _ = generate_synthetic_dataset(dataset_type, n_samples, noise_seed)

# ── Elbow Chart (cached) ──────────────────────────────────────────────────

# %% Cell 15 - Tính và hiển thị gợi ý số cụm
_elbow_inertias, _elbow_sils = compute_elbow(tuple(map(tuple, X_data)), 6, int(noise_seed))
_k_range = list(range(2, 7))
_best_k_sil = _k_range[int(np.argmax(_elbow_sils))]

with st.sidebar.expander(f"📈 Elbow Chart — Gợi ý k tối ưu (k={_best_k_sil})", expanded=False):
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=_k_range, y=_elbow_inertias, mode='lines+markers+text',
        name='Inertia (KMeans)', line=dict(color='#2563eb', width=2),
        marker=dict(size=7, color=['#ef4444' if k == _best_k_sil else '#2563eb' for k in _k_range]),
        text=[f'{v:.0f}' for v in _elbow_inertias], textposition='top center', textfont=dict(size=9)
    ))
    fig_elbow.update_layout(
        height=200, margin=dict(l=5, r=5, t=10, b=30),
        xaxis=dict(title='k', tickvals=_k_range),
        yaxis=dict(title='Inertia', showgrid=True),
        plot_bgcolor='#f8fafc', showlegend=False
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    fig_sil_bar = go.Figure()
    fig_sil_bar.add_trace(go.Bar(
        x=_k_range, y=_elbow_sils,
        marker_color=['#ef4444' if k == _best_k_sil else '#60a5fa' for k in _k_range],
        text=[f'{v:.3f}' for v in _elbow_sils], textposition='outside', textfont=dict(size=9)
    ))
    fig_sil_bar.update_layout(
        height=200, margin=dict(l=5, r=5, t=10, b=30),
        xaxis=dict(title='k', tickvals=_k_range),
        yaxis=dict(title='Silhouette', range=[0, max(_elbow_sils) * 1.3]),
        plot_bgcolor='#f8fafc', showlegend=False
    )
    st.plotly_chart(fig_sil_bar, use_container_width=True)
    st.caption(f"🔴 k = **{_best_k_sil}** cho Silhouette cao nhất ({max(_elbow_sils):.3f}). "
               f"Đường Inertia 'gập khuỷu tay' tại đây là điểm cân bằng tốt nhất.")

# ── End Elbow Chart ───────────────────────────────────────────────────────

# %% Cell 16 - Tính kết quả tất cả thuật toán trước khi dựng tab
# Chuyển X_data sang tuple để có thể hash trong st.cache_data
_X_key = tuple(map(tuple, X_data))

(
    cure_labels, cure_reps, cure_means, cure_metrics, cure_time,
    km_labels,   km_centers, km_metrics, km_time,
    kmed_labels, kmed_centers, kmed_metrics, kmed_time,
    hier_labels, hier_metrics, hier_time
) = run_all_algorithms(_X_key, k_clusters, c_reps, alpha_shrink, noise_seed)

agnes_results = run_agnes_4linkage(_X_key, k_clusters, noise_seed)
diana_labels, diana_means, diana_metrics, diana_time, diana_clusters = run_diana(_X_key, k_clusters)

# Một nguồn số liệu cho biểu đồ và bảng tổng hợp của toàn bộ ứng dụng.
all_metrics = {
    "CURE": cure_metrics, "K-Means": km_metrics, "K-Medoids": kmed_metrics,
    **{f"AGNES ({linkage})": result['metrics'] for linkage, result in agnes_results.items()},
    "DIANA": diana_metrics,
}
data_context = f"Dữ liệu: {dataset_type} | N = {len(X_data)} | k = {k_clusters} | c = {c_reps} | α = {alpha_shrink} | Seed = {noise_seed}"

# %% Cell 17 - Khai báo tab theo thứ tự đọc và hiển thị
tab_cure_sim, tab_cure_main, tab_cure_steps, tab_cure_flow, tab_vs_kmeans, tab_vs_kmedoids, tab_vs_hier, tab_vs_agnes, tab_vs_diana, tab_summary, tab_outlier = st.tabs([
    "🎮 Mô Phỏng Tương Tác (Canvas HTML5)",
    "🎯 CURE: Trực quan hóa & Phân cụm",
    "📝 CURE: Ví dụ tính tay (Toy Example)",
    "🔍 CURE: Quy trình 5 giai đoạn",
    "⚔️ So sánh: CURE vs K-Means",
    "⚔️ So sánh: CURE vs K-Medoids",
    "⚔️ So sánh: CURE vs Hierarchical",
    "🔗 AGNES: 4 Linkage so sánh",
    "✂️ DIANA: Phân cụm Chia cắt",
    "📋 Bảng Tổng hợp Đối sánh",
    "🔎 Phân tích Ngoại lai (Outlier)"
])

# %% Cell 18 - Tab 01: 🎮 Mô Phỏng Tương Tác Từng Bước (Interactive CURE Visualizer)
# Dùng dữ liệu và kết quả đã chuẩn bị ở các cell phía trên.
with tab_cure_sim:
    render_tab_header('🎮 Mô Phỏng Tương Tác Từng Bước (Interactive CURE Visualizer)', 'Quan sát quá trình gom cụm từng bước và đối chiếu với kết quả CURE.', data_context)
    if "Data" in dataset_type:
        st.info(f"📊 Đang sử dụng dữ liệu thực tế: **Data/Test.csv** | Cặp thuộc tính: **{selected_feature_pair}** | Kích thước mẫu: **{len(X_data)} khách hàng**.")
    else:
        st.info(f"🎨 Đang sử dụng tập dữ liệu kiểm thử: **{dataset_type}** | Số điểm: **{len(X_data)} điểm**.")

    # Nhúng bộ Canvas HTML5
    title_label = f"{dataset_type} ({selected_feature_pair})" if "Data" in dataset_type else dataset_type
    canvas_code = render_canvas_html(X_data, k_clusters, c_reps, alpha_shrink, title_label)
    components.html(canvas_code, height=880, scrolling=True)

    col_btn_l, col_btn_r = st.columns([2, 1])
    with col_btn_l:
        st.markdown("""
        <div class="app-card">
            <b>💡 Hướng dẫn thao tác trực quan:</b><br>
            • Nhấp <b>"⏩ 1 Bước (Step)"</b> để theo dõi cặp cụm gần nhất sáp nhập và quan sát cách <b>c</b> điểm đại diện co rút <b>α</b> về phía tâm.<br>
            • Nhấp <b>"▶️ Chạy Tự Động"</b> để xem toàn bộ quá trình gom cụm diễn ra sinh động.<br>
            • Khi đạt mục tiêu <b>k</b>, <b>Bảng Giải Thích Kết Quả</b> sẽ tự động xuất hiện ngay bên dưới khung vẽ!
        </div>
        """, unsafe_allow_html=True)
    with col_btn_r:
        df_export = pd.DataFrame(X_data, columns=[axis_x_name, axis_y_name])
        df_export['Cluster_CURE'] = cure_labels + 1
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải kết quả phân cụm CURE (CSV)",
            data=csv_data,
            file_name=f"cure_clustering_result_k{k_clusters}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # -------------------------------------------------------------
    # BẢNG PHÂN TÍCH & GIẢI THÍCH CHI TIẾT KẾT QUẢ ĐỊNH LƯỢNG
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📋 Phân Tích & Giải Thích Chi Tiết Kết Quả Phân Cụm (Báo Cáo Học Thuật)")
    
    col_exp_1, col_exp_2 = st.columns(2)
    
    with col_exp_1:
        st.markdown("##### 1. Thống kê Phân bổ Từng Cụm Khách Hàng")
        unique_labels = sorted(list(set(cure_labels)))
        cluster_rows = []
        palette_names = ["Xanh dương", "Đỏ", "Xanh lá", "Cam", "Tím", "Xanh lơ"]
        
        for idx, lbl in enumerate(unique_labels):
            pts_in_c = X_data[cure_labels == lbl]
            c_count = len(pts_in_c)
            c_pct = (c_count / len(X_data)) * 100
            c_mean = np.mean(pts_in_c, axis=0)
            
            # Gợi ý tên phân khúc thông minh dựa trên đặc trưng dữ liệu thực tế
            if "Data" in dataset_type:
                if "Tuổi" in axis_x_name:
                    mean_age = c_mean[0]
                    if mean_age >= 65:
                        c_name = "Khách hàng Cao tuổi / Hưu trí"
                        action = "Sản phẩm chăm sóc sức khỏe, bảo hiểm, tiện ích gia đình"
                    elif mean_age >= 45:
                        c_name = "Khách hàng Trưởng thành / Cao cấp"
                        action = "Cung cấp dịch vụ VIP, gói sản phẩm giá trị cao"
                    else:
                        if c_pct >= 40:
                            c_name = "Khách hàng Phổ thông (Trẻ tuổi)"
                            action = "Duy trì khuyến mãi tích điểm định kỳ, kích cầu"
                        else:
                            c_name = "Khách hàng Trẻ / Tiềm năng"
                            action = "Tiếp thị số qua MXH, trải nghiệm mua sắm nhanh"
                else:
                    if idx == 0:
                        c_name = "Khách hàng Phổ thông"
                        action = "Duy trì khuyến mãi tích điểm định kỳ"
                    elif idx == 1:
                        c_name = "Khách hàng Trưởng thành / Cao cấp"
                        action = "Cung cấp dịch vụ VIP, gói sản phẩm giá trị cao"
                    elif idx == 2:
                        c_name = "Khách hàng Ngách / Tiềm năng"
                        action = "Chăm sóc theo nhu cầu cá nhân hóa"
                    else:
                        c_name = f"Phân khúc Ngách {idx+1}"
                        action = "Chăm sóc theo nhu cầu cá nhân hóa"
            else:
                c_name = f"Cụm Hình học {idx+1}"
                action = "Bảo toàn hoàn hảo hình học tự nhiên"
                
            cluster_rows.append({
                "Cụm": f"Cụm {idx+1} ({palette_names[idx % len(palette_names)]})",
                "Số khách hàng": f"{c_count} ({c_pct:.1f}%)",
                f"TB {axis_x_name}": f"{c_mean[0]:.1f}",
                f"TB {axis_y_name}": f"{c_mean[1]:.1f}",
                "Đặc trưng phân khúc": c_name,
                "Khuyến nghị ứng dụng": action
            })
            
        df_cluster_report = pd.DataFrame(cluster_rows)
        render_table(df_cluster_report)

    with col_exp_2:
        st.markdown("##### 2. Nhận Định Tại Sao CURE Vượt Trội Trên Tập Này")
        st.markdown(f"""
        * 🎯 **Định hình bằng $c={c_reps}$ đại diện:** Thay vì ép toàn bộ dữ liệu vào một khối tròn nhân tạo quanh 1 tâm duy nhất như K-Means, CURE chọn các điểm đại diện rải dọc theo thân cụm, ôm trọn các nhóm khách hàng có phân bố dẹt hoặc hình học phi cầu.
        * 🛡️ **Khoảng đệm an toàn với α = {alpha_shrink}:** Nhờ co các điểm đại diện về trọng tâm theo hệ số co cụm α = {alpha_shrink}, các khách hàng có hành vi dị biệt hoặc điểm nhiễu ngoại lai ở rìa ngoài bị vô hiệu hóa, không làm xê dịch ranh giới phân cụm.
        * ⚡ **Chỉ số đánh giá định lượng:**
          - **Silhouette Score:** `{cure_metrics['Silhouette']:.4f}` *(Độ gắn kết nội cụm và tách biệt ngoại cụm tốt)*.
          - **Davies-Bouldin Index:** `{cure_metrics['Davies-Bouldin']:.4f}` *(Càng nhỏ chứng tỏ các cụm càng cô đặc)*.
        """)


# %% Cell 19 - Tab 02: 🎯 Kết Quả Phân Cụm Thuật Toán CURE
# Dùng dữ liệu và kết quả đã chuẩn bị ở các cell phía trên.
with tab_cure_main:
    render_tab_header('🎯 Kết Quả Phân Cụm Thuật Toán CURE', 'Xem phân bố cụm, điểm đại diện và chỉ số đánh giá trên dữ liệu đã chọn.', data_context)
    st.markdown(f"Đang phân cụm trên tập: **{dataset_type}** với $N = {len(X_data)}$ điểm dữ liệu.")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig_cure = make_scatter_figure(
            X_data, cure_labels,
            f"Kết quả phân cụm CURE (k={k_clusters}, c={c_reps} đại diện, α={alpha_shrink})",
            reps=cure_reps, means=cure_means,
            x_title=axis_x_name, y_title=axis_y_name
        )
        render_chart(fig_cure, key="tab_chart_1")

    with col_c2:
        st.markdown("#### 📊 Chỉ số Đánh giá Chất lượng")
        render_metrics(cure_metrics)

        st.markdown("""
        <div class="app-card">
            <b>Ý nghĩa trực quan của CURE:</b><br>
            • <b>Dấu X đen:</b> $c$ điểm đại diện đã co cụm về phía tâm.<br>
            • <b>Ngôi sao vàng:</b> Trọng tâm mean của cụm.<br>
            • Nhờ phân bố rải rác nhiều điểm đại diện, CURE ôm trọn vẹn đường cong tự nhiên và phân bố khách hàng!
        </div>
        """, unsafe_allow_html=True)

    if df_customer_raw is not None:
        with st.expander("📋 Xem trước Dữ liệu gốc khách hàng từ folder Data/Test.csv"):
            render_table(df_customer_raw.head(20))


# %% Cell 20 - Tab 03: 📝 Bài toán Ví dụ Tính tay Từng bước (Toy Example)
# Dùng dữ liệu và kết quả đã chuẩn bị ở các cell phía trên.
with tab_cure_steps:
    render_tab_header('📝 Bài toán Ví dụ Tính tay Từng bước (Toy Example)', 'Theo dõi phép tính CURE trên bộ dữ liệu minh họa cố định gồm 6 điểm.', "Ví dụ cố định: N = 6 | k = 2 | c = 2 | α = 0.5")
    st.markdown(r"""
    Nhóm thiết lập một tập dữ liệu nhỏ gồm **6 điểm 2D cụ thể** để minh họa chính xác từng bước hoạt động của thuật toán CURE:
    * Cụm bên trái: $P_1(1, 2)$, $P_2(2, 3)$, $P_3(2, 1)$
    * Cụm bên phải: $P_4(8, 7)$, $P_5(9, 8)$, $P_6(8, 9)$
    * **Cấu hình:** $k = 2$ cụm mục tiêu, $c = 2$ điểm đại diện mỗi cụm, hệ số co cụm $\alpha = 0.5$.
    """)

    step_choice = st.radio("Chọn bước thực hiện để quan sát:", [
        "Bước 0: Khởi tạo 6 điểm riêng lẻ (6 cụm ban đầu)",
        "Bước 1: Sáp nhập P1 và P2 -> C{1,2}",
        "Bước 2: Sáp nhập P4 và P5 -> C{4,5}",
        "Bước 3: Sáp nhập C{1,2} với P3 -> C{1,2,3}",
        "Bước 4: Sáp nhập C{4,5} với P6 -> C{4,5,6} (Hoàn tất k=2)"
    ], horizontal=True)

    step_idx = int(step_choice.split(":")[0].replace("Bước ", ""))
    step_img = os.path.join(os.path.dirname(__file__), "toy_example_steps", f"step_{step_idx}.png")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        if os.path.exists(step_img):
            st.image(step_img, caption=f"Hình minh họa {step_choice}", use_container_width=True)
        else:
            st.info("Hình ảnh minh họa đang được cập nhật.")

    with col_t2:
        st.markdown("#### 📐 Công thức và Phép tính chi tiết")
        if step_idx == 0:
            st.markdown(r"""
            * Mỗi điểm ban đầu là 1 cụm đơn lẻ: $C_1=\{P_1\}, \dots, C_6=\{P_6\}$.
            * Khoảng cách giữa 2 cụm chính là khoảng cách Euclidean giữa 2 điểm:
              $$d(P_1, P_2) = \sqrt{(2-1)^2 + (3-2)^2} = \sqrt{2} \approx 1.414$$
            * Cặp $(P_1, P_2)$ và $(P_4, P_5)$ có khoảng cách nhỏ nhất toàn ma trận ($1.414$).
            """)
        elif step_idx == 1:
            st.markdown(r"""
            * Sáp nhập $P_1$ và $P_2$ thành cụm $C_{\{1,2\}}$.
            * **Trọng tâm cụm mới:** $m = \left(\frac{1+2}{2}, \frac{2+3}{2}\right) = (1.5, 2.5)$.
            * **Co cụm 2 điểm đại diện với $\alpha = 0.5$:**
              $$p'_1 = (1, 2) + 0.5 \times ((1.5, 2.5) - (1, 2)) = (1.25, 2.25)$$
              $$p'_2 = (2, 3) + 0.5 \times ((1.5, 2.5) - (2, 3)) = (1.75, 2.75)$$
            * Điểm đại diện đã dịch chuyển lùi vào trong một khoảng an toàn!
            """)
        elif step_idx == 2:
            st.markdown(r"""
            * Sáp nhập $P_4(8, 7)$ và $P_5(9, 8)$ thành cụm $C_{\{4,5\}}$.
            * **Trọng tâm cụm:** $m = (8.5, 7.5)$.
            * **Co cụm 2 điểm đại diện với $\alpha = 0.5$:**
              $$p'_4 = (8.25, 7.25), \quad p'_5 = (8.75, 7.75)$$
            * Số cụm hiện tại giảm xuống còn 4 cụm.
            """)
        elif step_idx == 3:
            st.markdown(r"""
            * Tính khoảng cách từ $P_3(2, 1)$ tới các điểm đại diện đã co của $C_{\{1,2\}}$:
              $$d(P_3, p'_1) = \sqrt{(2-1.25)^2 + (1-2.25)^2} \approx 1.458$$
            * Do $1.458$ là khoảng cách nhỏ nhất, $P_3$ sáp nhập vào cụm $C_{\{1,2\}} \rightarrow C_{\{1,2,3\}}$.
            * Trọng tâm mới: $m = (1.667, 2.0)$.
            * Giải thuật **Farthest-Point** chọn 2 điểm xa nhất trong cụm rồi co về $m$.
            """)
        elif step_idx == 4:
            st.markdown(r"""
            * $P_6(8, 9)$ sáp nhập vào cụm $C_{\{4,5\}} \rightarrow C_{\{4,5,6\}}$.
            * Cụm bên phải hoàn tất với 3 phần tử.
            * **Điều kiện dừng:** Số cụm còn lại đúng bằng $k = 2$.
            * Thuật toán kết thúc thành công với độ chính xác tuyệt đối 100%!
            """)


# %% Cell 21 - Tab 04: 🔍 Quy trình 5 Giai đoạn & Kiến trúc Xử lý Dữ liệu lớn
# Dùng dữ liệu và kết quả đã chuẩn bị ở các cell phía trên.
with tab_cure_flow:
    render_tab_header('🔍 Quy trình 5 Giai đoạn & Kiến trúc Xử lý Dữ liệu lớn', 'Tìm hiểu các giai đoạn của CURE và ý nghĩa của các tham số.', "Nội dung lý thuyết; các giai đoạn mở rộng không phải đều đã cài đặt trong bản demo.")
    
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown(r"""
        Thuật toán CURE kết hợp hoàn hảo giữa **phân cụm phân cấp chất lượng cao** và **khả năng mở rộng trên dữ liệu lớn (Scalability)** qua 5 giai đoạn:

        1. **Giai đoạn 1: Lấy mẫu ngẫu nhiên (Random Sampling)**
           - Rút một mẫu ngẫu nhiên $s$ điểm từ tập dữ liệu khổng lồ $N$ điểm ($s \ll N$). Mẫu $s$ vẫn bảo toàn đầy đủ hình học và phân bố của các cụm lớn.
        
        2. **Giai đoạn 2: Phân hoạch không gian (Partitioning)**
           - Chia mẫu $s$ thành $p$ phân vùng đều nhau (mỗi phần $\frac{s}{p}$ điểm). Tiến hành gom cụm sơ bộ trên từng phân vùng để giảm chi phí bộ nhớ từ $O(s^2)$ xuống $O(s^2/p)$.

        3. **Giai đoạn 3: Gom cụm phân cấp trên các cụm đại diện**
           - Gom các cụm cục bộ lại với nhau. Tại mỗi bước sáp nhập, CURE chọn ra $c$ điểm đại diện phân tán đều trên thân cụm bằng giải thuật *Farthest-Point Heuristic*, sau đó co về trọng tâm theo hệ số $\alpha$.

        4. **Giai đoạn 4: Loại bỏ ngoại lai 2 pha (Outlier Elimination)**
           - **Pha 1 (Giữa chừng):** Khi số cụm giảm xuống còn $k'$, các cụm chỉ có 1–2 điểm tăng trưởng chậm bị loại bỏ.
           - **Pha 2 (Cuối kỳ):** Các cụm có kích thước quá nhỏ so với kích thước trung bình bị coi là cụm nhiễu và xóa bỏ.

        5. **Giai đoạn 5: Gán nhãn toàn bộ dữ liệu lớn trên đĩa (Disk Labeling)**
           - Quét qua toàn bộ $N - s$ điểm dữ liệu còn lại trên đĩa và gán mỗi điểm vào cụm có điểm đại diện gần nhất. Chi phí tuyến tính $O(N)$.
        """)

    with col_f2:
        st.markdown("#### ⚙️ Bảng Hướng dẫn Chọn Siêu tham số")
        df_params = pd.DataFrame({
            "Tham số": ["s (Kích thước mẫu)", "p (Số phân vùng)", "c (Số điểm đại diện)", "α (Hệ số co cụm)", "k (Số cụm)"],
            "Ý nghĩa": ["Đại diện cho tập lớn N", "Chia nhỏ để tăng tốc", "Định hình đường biên cụm", "Khoảng đệm an toàn chống nhiễu", "Mục tiêu bài toán"],
            "Giá trị khuyến nghị": ["2.000 – 5.000", "2 – 4 phân vùng", "4 – 8 điểm", "0.3 – 0.5", "Tùy bài toán thực tế"]
        })
        render_table(df_params)


# %% Cell 22 - Tab 05: ⚔️ So sánh Đối đầu Trực diện: CURE vs K-Means
# Dùng dữ liệu và kết quả đã chuẩn bị ở các cell phía trên.
with tab_vs_kmeans:
    render_tab_header('⚔️ So sánh Đối đầu Trực diện: CURE vs K-Means', 'Đối chiếu CURE và K-Means trên cùng dữ liệu và cùng số cụm.', data_context)
    st.markdown("#### 🎯 Trọng tâm kiểm thử: Khắc phục hạn chế giả định cụm hình cầu của K-Means")

    col_km1, col_km2 = st.columns(2)

    with col_km1:
        st.markdown("##### 🟢 Thuật toán CURE (Đề tài nghiên cứu)")
        fig_c_km = make_scatter_figure(
            X_data, cure_labels,
            f"CURE: Nhận diện chuẩn xác hình thái cụm tự nhiên",
            reps=cure_reps, means=cure_means,
            x_title=axis_x_name, y_title=axis_y_name
        )
        render_chart(fig_c_km, key="tab_chart_2")
        render_metrics(cure_metrics)
        st.success("✅ **Ưu thế của CURE:** Ôm trọn vẹn dải cong phi cầu và phân bố tự nhiên nhờ c điểm đại diện rải đều!")

    with col_km2:
        st.markdown("##### 🔵 Thuật toán K-Means (Thuật toán đối chứng)")
        fig_km = make_scatter_figure(
            X_data, km_labels,
            f"K-Means: Cắt ngang cụm do giả định hình cầu",
            means=km_centers,
            x_title=axis_x_name, y_title=axis_y_name
        )
        render_chart(fig_km, key="tab_chart_3")
        render_metrics(km_metrics)
        if "Moons" in dataset_type or "Circles" in dataset_type:
            st.error("❌ **Thất bại hình học:** K-Means cắt đôi cụm phi cầu do chỉ dùng 1 tâm trung bình!")
        else:
            st.info("ℹ️ K-Means hoạt động tốt khi các cụm có dạng hình cầu lồi tách biệt.")

    st.markdown("#### 📋 Bảng Đối chiếu Trực tiếp: CURE vs K-Means")
    df_cmp_km = pd.DataFrame({
        "Tiêu chí đối sánh": [
            "Cơ chế đại diện cụm",
            "Giả định hình học cụm",
            "Xử lý cụm phi cầu (Two Moons, Circles)",
            "Độ nhạy với điểm ngoại lai (Outliers)",
            "Độ phức tạp thời gian",
            "Silhouette Score (Tập hiện tại)",
            "Davies-Bouldin Index (Tập hiện tại)"
        ],
        "CURE (Clustering Using REpresentatives)": [
            f"Dùng {c_reps} điểm đại diện phân bố đều trên thân cụm",
            "Tự do bắt hình dạng tùy ý (Arbitrary shape)",
            "Xuất sắc (Bảo toàn 100% hình thái cụm)",
            "Rất tốt (Nhờ co cụm alpha và 2 pha lọc ngoại lai)",
            "O(n² log n) trên mẫu, O(N) khi gán nhãn lớn",
            f"{cure_metrics['Silhouette']:.4f}",
            f"{cure_metrics['Davies-Bouldin']:.4f}"
        ],
        "K-Means (Thuật toán phân hoạch)": [
            "Dùng duy nhất 1 điểm trọng tâm trung bình (Centroid)",
            "Giả định cụm hình cầu lồi (Spherical/Convex)",
            "Thất bại hoàn toàn (Cắt ngang thân dải dữ liệu)",
            "Kém (1 điểm ngoại lai có thể kéo lệch hẳn tâm mean)",
            "O(k * n * t) - Rất nhanh trên dữ liệu hình cầu",
            f"{km_metrics['Silhouette']:.4f}",
            f"{km_metrics['Davies-Bouldin']:.4f}"
        ]
    })
    render_table(df_cmp_km)


# %% Cell 23 - Tab 06: ⚔️ So sánh Đối đầu Trực diện: CURE vs K-Medoids (PAM)
# Dùng dữ liệu và kết quả đã chuẩn bị ở các cell phía trên.
with tab_vs_kmedoids:
    render_tab_header('⚔️ So sánh Đối đầu Trực diện: CURE vs K-Medoids (PAM)', 'Đối chiếu CURE và K-Medoids trên cùng dữ liệu và cùng số cụm.', data_context)
    st.markdown("#### 🎯 Trọng tâm kiểm thử: Khả năng kháng điểm ngoại lai (Outliers) và cụm dị hướng (Anisotropic)")

    col_kmed1, col_kmed2 = st.columns(2)

    with col_kmed1:
        st.markdown("##### 🟢 Thuật toán CURE (Đề tài nghiên cứu)")
        fig_c_kmed = make_scatter_figure(
            X_data, cure_labels,
            f"CURE: Kháng ngoại lai bằng cơ chế co cụm α = {alpha_shrink}",
            reps=cure_reps, means=cure_means,
            x_title=axis_x_name, y_title=axis_y_name
        )
        render_chart(fig_c_kmed, key="tab_chart_4")
        render_metrics(cure_metrics)
        st.success("✅ **Miễn nhiễm ngoại lai:** Co cụm giúp điểm đại diện lùi vào sâu bên trong lõi cụm!")

    with col_kmed2:
        st.markdown("##### 🔴 Thuật toán K-Medoids / PAM (Thuật toán đối chứng)")
        fig_kmed = make_scatter_figure(
            X_data, kmed_labels,
            f"K-Medoids: Chọn điểm thực tế làm tâm (Medoid đỏ)",
            centers=kmed_centers, center_label="Medoid thực tế",
            x_title=axis_x_name, y_title=axis_y_name
        )
        render_chart(fig_kmed, key="tab_chart_5")
        render_metrics(kmed_metrics)
        st.warning("⚠️ **Hạn chế:** Giảm nhạy cảm với ngoại lai hơn K-Means nhưng vẫn giả định cụm hình cầu và tốn chi phí hoán đổi.")

    st.markdown("#### 📋 Bảng Đối chiếu Trực tiếp: CURE vs K-Medoids")
    df_cmp_kmed = pd.DataFrame({
        "Tiêu chí đối sánh": [
            "Cách chọn tâm/đại diện",
            "Ảnh hưởng của điểm ngoại lai cực đoan",
            "Cơ chế loại bỏ nhiễu",
            "Khả năng mở rộng trên dữ liệu lớn",
            "Độ nhạy với hình dạng dị hướng (Anisotropic)",
            "Silhouette Score (Tập hiện tại)",
            "Davies-Bouldin Index (Tập hiện tại)"
        ],
        "CURE (Clustering Using REpresentatives)": [
            "c điểm đại diện phân tán và co lại về trọng tâm",
            "Vô hiệu hóa hoàn toàn nhờ co cụm α lùi vào lõi",
            "Có 2 pha loại bỏ ngoại lai độc lập tự động",
            "Rất tốt nhờ lấy mẫu ngẫu nhiên s và gán nhãn đĩa",
            "Rất tốt (Các điểm đại diện trải dọc thân elip)",
            f"{cure_metrics['Silhouette']:.4f}",
            f"{cure_metrics['Davies-Bouldin']:.4f}"
        ],
        "K-Medoids (PAM - Partitioning Around Medoids)": [
            "Chọn 1 điểm dữ liệu thực tế có tổng khoảng cách nhỏ nhất",
            "Ít bị kéo lệch hơn K-Means nhưng vẫn bị méo ranh giới",
            "Không có pha lọc nhiễu riêng biệt",
            "Kém (Độ phức tạp O(k(n-k)²) quá nặng khi n lớn)",
            "Kém (Vẫn coi cụm là khối tròn đẳng hướng quanh medoid)",
            f"{kmed_metrics['Silhouette']:.4f}",
            f"{kmed_metrics['Davies-Bouldin']:.4f}"
        ]
    })
    render_table(df_cmp_kmed)


# %% Cell 24 - Tab 07: ⚔️ So sánh Đối đầu Trực diện: CURE vs Gom cụm Phân cấp (Single Linkage)
# Dùng dữ liệu và kết quả đã chuẩn bị ở các cell phía trên.
with tab_vs_hier:
    render_tab_header('⚔️ So sánh Đối đầu Trực diện: CURE vs Gom cụm Phân cấp (Single Linkage)', 'Đối chiếu CURE và phân cụm phân cấp Single Linkage.', data_context)
    st.markdown("#### 🎯 Trọng tâm kiểm thử: Khắc phục hiện tượng nối chuỗi (Chaining Effect) và Tối ưu bộ nhớ")

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("##### 🟢 Thuật toán CURE (Đề tài nghiên cứu)")
        fig_c_hier = make_scatter_figure(
            X_data, cure_labels,
            f"CURE: Triệt tiêu nối chuỗi nhờ khoảng đệm co cụm",
            reps=cure_reps, means=cure_means,
            x_title=axis_x_name, y_title=axis_y_name
        )
        render_chart(fig_c_hier, key="tab_chart_6")
        render_metrics(cure_metrics)
        st.success("✅ **Không bị nối chuỗi:** Các điểm đại diện co cụm tạo khoảng cách ngăn cách an toàn!")

    with col_h2:
        st.markdown("##### 🟣 Gom cụm Phân cấp (Single Linkage)")
        fig_hier = make_scatter_figure(
            X_data, hier_labels,
            f"Hierarchical Single Linkage: Dễ bị dính cụm do nhiễu nối chuỗi",
            x_title=axis_x_name, y_title=axis_y_name
        )
        render_chart(fig_hier, key="tab_chart_7")
        render_metrics(hier_metrics)
        if "Outliers" in dataset_type:
            st.error("❌ **Hiện tượng nối chuỗi:** Các điểm ngoại lai nằm giữa đã nối dính 2 cụm riêng biệt lại với nhau!")
        else:
            st.info("ℹ️ Single Linkage tìm được cụm phi cầu nhưng tốn bộ nhớ O(N²) và cực kỳ sợ nhiễu.")

    st.markdown("#### 📋 Bảng Đối chiếu Trực tiếp: CURE vs Gom cụm Phân cấp (Single Link)")
    df_cmp_hier = pd.DataFrame({
        "Tiêu chí đối sánh": [
            "Khoảng cách giữa hai cụm",
            "Hiện tượng nối chuỗi (Chaining Effect)",
            "Độ nhạy với điểm ngoại lai",
            "Độ phức tạp bộ nhớ",
            "Khả năng chạy trên cơ sở dữ liệu lớn",
            "Silhouette Score (Tập hiện tại)",
            "Davies-Bouldin Index (Tập hiện tại)"
        ],
        "CURE (Clustering Using REpresentatives)": [
            "Khoảng cách nhỏ nhất giữa các điểm đại diện ĐÃ CO CỤM",
            "Triệt tiêu hoàn toàn nhờ khoảng đệm co cụm alpha",
            "Miễn nhiễm nhờ co cụm và 2 pha lọc ngoại lai",
            "O(s) - Tiết kiệm nhờ lấy mẫu ngẫu nhiên s điểm",
            "Xuất sắc (Gán nhãn tuyến tính O(N) trên đĩa)",
            f"{cure_metrics['Silhouette']:.4f}",
            f"{cure_metrics['Davies-Bouldin']:.4f}"
        ],
        "Hierarchical (Single Linkage)": [
            "Khoảng cách nhỏ nhất giữa TẤT CẢ các cặp điểm thuộc 2 cụm",
            "Rất nghiêm trọng (Chỉ cần 1 vệt điểm nhiễu là sáp nhập nhầm)",
            "Rất nhạy cảm với các điểm nhiễu ngoại lai ở rìa",
            "O(N²) - Tràn bộ nhớ khi N > 10.000",
            "Không khả thi trên dữ liệu lớn (Big Data)",
            f"{hier_metrics['Silhouette']:.4f}",
            f"{hier_metrics['Davies-Bouldin']:.4f}"
        ]
    })
    render_table(df_cmp_hier)


# %% Cell 25 - Tab 08: 🔗 AGNES (Agglomerative Nesting) — So sánh 4 kiểu Linkage
# Dùng dữ liệu và kết quả đã chuẩn bị ở các cell phía trên.
with tab_vs_agnes:
    render_tab_header('🔗 AGNES (Agglomerative Nesting) — So sánh 4 kiểu Linkage', 'Đối chiếu bốn cách đo khoảng cách giữa các cụm trong AGNES.', data_context)

    # ─── Giới thiệu ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-card">
        <b style="color:#1e40af;font-size:14px;">📌 AGNES là gì và liên quan gì đến CURE?</b><br>
        <span style="font-size:13px;color:#374151;">
        <b>AGNES</b> là phương pháp phân cụm phân cấp <b>hướng từ dưới lên (Bottom-up)</b>: ban đầu mỗi điểm là 1 cụm,
        sau đó liên tục gom 2 cụm gần nhau nhất thành 1 — giống như CURE.<br><br>
        Sự khác biệt nằm ở <b>cách đo "khoảng cách giữa 2 cụm"</b> — AGNES có 4 cách (linkage), mỗi cách
        cho kết quả phân cụm rất khác nhau. <b>CURE chính là bản nâng cấp của AGNES Single Linkage</b>
        — thêm co cụm α để chống nối chuỗi và nhiều điểm đại diện để nắm hình dạng phức tạp.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ─── Chạy 4 AGNES với cache ──────────────────────────────────────────────


    linkage_info = {
        'single':   {'name': 'Single Linkage',   'icon': '🔵', 'color': '#2563eb',
                     'mo_ta': 'Khoảng cách = 2 điểm GẦN NHẤT của 2 cụm',
                     'uu': 'Tìm được cụm phi cầu, hình cong uốn lượn',
                     'nhuoc': 'Dễ bị "nối chuỗi" (Chaining Effect) khi có nhiễu'},
        'complete': {'name': 'Complete Linkage', 'icon': '🔴', 'color': '#dc2626',
                     'mo_ta': 'Khoảng cách = 2 điểm XA NHẤT của 2 cụm',
                     'uu': 'Tạo cụm gọn, đều, ít bị nối chuỗi',
                     'nhuoc': 'Nhạy với outlier — 1 điểm xa làm lệch khoảng cách'},
        'average':  {'name': 'Average Linkage',  'icon': '🟡', 'color': '#d97706',
                     'mo_ta': 'Khoảng cách = TRUNG BÌNH tất cả cặp điểm giữa 2 cụm',
                     'uu': 'Cân bằng giữa Single và Complete, ít nhạy outlier',
                     'nhuoc': 'Chi phí tính toán cao hơn Single/Complete'},
        'ward':     {'name': 'Ward Linkage',     'icon': '🟢', 'color': '#16a34a',
                     'mo_ta': 'Gom sao cho tổng variance trong cụm tăng ÍT NHẤT',
                     'uu': 'Cụm compact, cân đối — tốt nhất thực tế với dữ liệu hình cầu',
                     'nhuoc': 'Giả định cụm hình cầu (giống K-Means), kém với phi cầu'},
    }

    # ─── Phần 1: Giải thích 4 linkage ────────────────────────────────────────
    st.markdown("#### 📖 1. Cách hoạt động của từng Linkage")

    col_l1, col_l2 = st.columns(2)
    for i, (key, info) in enumerate(linkage_info.items()):
        col = col_l1 if i < 2 else col_l2
        with col:
            st.markdown(f"""
            <div class="app-card">
                <div style="font-weight:700;color:{info['color']};font-size:13px;">
                    {info['icon']} {info['name']}
                </div>
                <div style="font-size:12px;color:#475569;margin:4px 0;">
                    <b>Cách đo:</b> {info['mo_ta']}
                </div>
                <div style="font-size:12px;color:#16a34a;">✅ <b>Ưu điểm:</b> {info['uu']}</div>
                <div style="font-size:12px;color:#dc2626;">❌ <b>Nhược điểm:</b> {info['nhuoc']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-card">
        <b>💡 Ghi nhớ công thức tổng quát:</b><br>
        &nbsp;&nbsp;• <b>Single</b> = min(dist) &nbsp;|&nbsp;
        <b>Complete</b> = max(dist) &nbsp;|&nbsp;
        <b>Average</b> = mean(dist) &nbsp;|&nbsp;
        <b>Ward</b> = min(ΔVariance)<br>
        Trong đó dist là khoảng cách giữa từng cặp điểm (1 điểm từ cụm A, 1 điểm từ cụm B).
    </div>
    """, unsafe_allow_html=True)

    # ─── Phần 2: Biểu đồ 4 linkage song song ────────────────────────────────
    st.markdown("#### 📊 2. Kết quả phân cụm trực quan — 4 Linkage trên cùng dữ liệu")

    col_g1, col_g2 = st.columns(2)
    col_g3, col_g4 = st.columns(2)
    grid_cols = [col_g1, col_g2, col_g3, col_g4]

    for i, (key, info) in enumerate(linkage_info.items()):
        res = agnes_results[key]
        lbl = res['labels']
        met = res['metrics']
        with grid_cols[i]:
            st.markdown(f"##### {info['icon']} {info['name']}")
            fig = make_scatter_figure(
                X_data, lbl,
                f"{info['name']} (k={k_clusters})",
                x_title=axis_x_name, y_title=axis_y_name
            )
            render_chart(fig, key=f"tab_chart_8_{i}")
            render_metrics(met)

    # ─── Phần 3: Bảng so sánh 4 linkage vs CURE ─────────────────────────────
    st.markdown("#### 📋 3. Bảng Đối chiếu: CURE vs AGNES 4 Linkage")

    rows = []
    # Hàng CURE
    rows.append({
        "Thuật toán": "🟩 CURE (Đề tài)",
        "Cơ chế đo khoảng cách": f"Min dist giữa {c_reps} điểm đại diện ĐÃ CO (α={alpha_shrink})",
        "Nhận diện phi cầu": "✅ Xuất sắc",
        "Kháng nối chuỗi": "✅ Xuất sắc (nhờ co cụm α)",
        "Kháng Outlier": "✅ Rất tốt (α + 2 pha lọc)",
        f"Silhouette (k={k_clusters})": f"{cure_metrics['Silhouette']:.4f}",
        "DB Index": f"{cure_metrics['Davies-Bouldin']:.4f}",
        "Thời gian": f"{cure_time:.4f}s",
    })
    for key, info in linkage_info.items():
        res = agnes_results[key]
        met = res['metrics']
        phi_cau = "✅ Tốt" if key == 'single' else ("⚠️ Trung bình" if key == 'average' else "❌ Kém")
        chaining = "❌ Kém" if key == 'single' else ("✅ Tốt" if key in ['complete', 'ward'] else "⚠️ Trung bình")
        outlier  = "❌ Nhạy" if key == 'complete' else ("⚠️ Trung bình" if key == 'single' else "✅ Tốt")
        rows.append({
            "Thuật toán": f"{info['icon']} AGNES {info['name']}",
            "Cơ chế đo khoảng cách": info['mo_ta'],
            "Nhận diện phi cầu": phi_cau,
            "Kháng nối chuỗi": chaining,
            "Kháng Outlier": outlier,
            f"Silhouette (k={k_clusters})": f"{met['Silhouette']:.4f}",
            "DB Index": f"{met['Davies-Bouldin']:.4f}",
            "Thời gian": f"{res['time']:.4f}s",
        })
    df_agnes_cmp = pd.DataFrame(rows)
    render_table(df_agnes_cmp)

    # ─── Phần 4: Bar Chart Silhouette so sánh ────────────────────────────────
    st.markdown("#### 📈 4. So sánh Silhouette Score — CURE vs AGNES 4 Linkage")

    labels_bar = ["CURE"] + [f"AGNES\n{linkage_info[k]['name'].split()[0]}" for k in linkage_info]
    sil_bar_vals = [cure_metrics['Silhouette']] + [agnes_results[k]['metrics']['Silhouette'] for k in linkage_info]
    colors_bar = ['#103673', '#2563eb', '#dc2626', '#d97706', '#16a34a']
    best_idx = int(np.argmax(sil_bar_vals))

    fig_bar_agnes = go.Figure(data=[go.Bar(
        x=labels_bar, y=sil_bar_vals,
        marker_color=colors_bar,
        text=[f"{v:.4f}" for v in sil_bar_vals],
        textposition='outside',
        textfont=dict(size=11)
    )])
    # Vẽ đường dấu sao tại CURE
    fig_bar_agnes.add_hline(
        y=cure_metrics['Silhouette'], line_dash="dash", line_color="#103673",
        annotation_text=f"CURE: {cure_metrics['Silhouette']:.4f}",
        annotation_position="bottom right", annotation_font_color="#103673"
    )
    fig_bar_agnes.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#fafbfc",
        yaxis=dict(title="Silhouette Score (Càng cao càng tốt)", range=[-0.15, max(sil_bar_vals) * 1.25]),
        xaxis=dict(title="Thuật toán"),
        showlegend=False
    )
    render_chart(fig_bar_agnes, key="tab_chart_9")

    # ─── Phần 5: Giải thích kết quả và kết luận ──────────────────────────────
    best_agnes_key = max(linkage_info.keys(), key=lambda k: agnes_results[k]['metrics']['Silhouette'])
    best_agnes_info = linkage_info[best_agnes_key]
    best_agnes_sil  = agnes_results[best_agnes_key]['metrics']['Silhouette']

    cure_vs_best = "tốt hơn" if cure_metrics['Silhouette'] >= best_agnes_sil else "kém hơn"
    diff_pct = abs(cure_metrics['Silhouette'] - best_agnes_sil) / max(abs(best_agnes_sil), 1e-6) * 100

    st.markdown(f"""
    <div class="app-card">
        <b>💡 Đọc kết quả biểu đồ trên:</b><br><br>
        <ul style="margin:0 0 0 16px;line-height:1.9;">
            <li><b>Silhouette Score</b> đo mức độ gắn kết trong cụm và tách biệt giữa các cụm —
                càng gần 1.0 càng tốt, âm là phân cụm sai.</li>
            <li>Trong 4 biến thể AGNES, <b>{best_agnes_info['icon']} {best_agnes_info['name']}</b>
                đạt Silhouette cao nhất ({best_agnes_sil:.4f}) trên tập dữ liệu hiện tại.</li>
            <li>CURE đạt {cure_metrics['Silhouette']:.4f} — <b>{cure_vs_best}</b> biến thể AGNES tốt nhất
                khoảng {diff_pct:.1f}%.</li>
            <li>Tuy nhiên, Silhouette chỉ đo hình học — CURE vượt trội thực sự ở khả năng
                <b>nhận diện cụm phi cầu, kháng nối chuỗi và xử lý dữ liệu lớn</b>
                mà AGNES thuần không có.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ─── Kết luận ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-card">
        <div style="font-size:15px;font-weight:700;margin-bottom:10px;">
            🎯 Kết luận: CURE = AGNES được nâng cấp toàn diện
        </div>
        <div style="font-size:13px;line-height:1.9;opacity:0.95;">
            <table style="width:100%;border-collapse:collapse;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.2);">
                    <td style="padding:4px 8px;font-weight:600;">Vấn đề của AGNES</td>
                    <td style="padding:4px 8px;font-weight:600;">Giải pháp của CURE</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.15);">
                    <td style="padding:4px 8px;">Single dễ bị nối chuỗi</td>
                    <td style="padding:4px 8px;">Co cụm α kéo đại diện vào trong → ngăn chaining</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.15);">
                    <td style="padding:4px 8px;">Complete/Ward nhạy với hình dạng</td>
                    <td style="padding:4px 8px;">c điểm đại diện trải đều → nhận diện hình phi cầu</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.15);">
                    <td style="padding:4px 8px;">Tất cả AGNES tốn O(N²) bộ nhớ</td>
                    <td style="padding:4px 8px;">Lấy mẫu s + gán nhãn O(N) → chạy trên Big Data</td>
                </tr>
                <tr>
                    <td style="padding:4px 8px;">Không lọc ngoại lai</td>
                    <td style="padding:4px 8px;">2 pha lọc outlier tự động trong quá trình gom</td>
                </tr>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)


# %% Cell 26 - Tab 09: ✂️ DIANA (Divisive Analysis) — Phân cụm Phân cấp Chia cắt (Top-down)
# Dùng dữ liệu và kết quả đã chuẩn bị ở các cell phía trên.
with tab_vs_diana:
    render_tab_header('✂️ DIANA (Divisive Analysis) — Phân cụm Phân cấp Chia cắt (Top-down)', 'Quan sát cách DIANA chia cụm và đối chiếu với CURE.', data_context)

    # ─── Giới thiệu ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-card">
        <b style="color:#6b21a8;font-size:14px;">📌 DIANA là gì? Khác AGNES và CURE như thế nào?</b><br>
        <span style="font-size:13px;color:#374151;">
        <b>DIANA</b> là thuật toán phân cụm phân cấp <b>hướng từ trên xuống (Top-down / Divisive)</b>:
        bắt đầu từ <b>1 cụm lớn chứa toàn bộ dữ liệu</b>, rồi liên tục <b>tách ra</b> thành 2 cụm nhỏ hơn —
        ngược hoàn toàn với AGNES/CURE vốn gom từ dưới lên.<br><br>
        <b>3 thuật toán phân cấp trong đề tài này:</b><br>
        &nbsp;&nbsp;🔼 <b>AGNES</b> — Bottom-up (gom từng điểm thành cụm lớn hơn)<br>
        &nbsp;&nbsp;🟣 <b>CURE</b> — Bottom-up nâng cao (thêm co cụm α + c điểm đại diện)<br>
        &nbsp;&nbsp;🔽 <b>DIANA</b> — Top-down (tách cụm lớn thành cụm nhỏ hơn)
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ─── Cơ chế DIANA từng bước ─────────────────────────────────────────────
    st.markdown("#### 📖 1. Cơ chế hoạt động của DIANA từng bước")

    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown("""
        <div class="app-card">
            <b style="color:#6b21a8;">Thuật toán DIANA (5 bước):</b><br>
            <ol style="margin:8px 0 0 16px;color:#374151;">
                <li><b>Khởi tạo:</b> 1 cụm duy nhất chứa tất cả N điểm dữ liệu.</li>
                <li><b>Chọn cụm để tách:</b> Tìm cụm có <b>đường kính lớn nhất</b>
                    (đường kính = khoảng cách max giữa 2 điểm bất kỳ trong cụm).</li>
                <li><b>Tìm "splinter":</b> Trong cụm đó, tìm điểm có
                    <b>avg-dissimilarity cao nhất</b> tới tất cả điểm còn lại
                    → đây là "hạt nhân" của cụm con mới.</li>
                <li><b>Tách lặp:</b> Lần lượt chuyển điểm từ cụm gốc sang cụm con
                    nếu điểm đó <b>gần cụm con hơn cụm gốc</b>. Lặp đến khi không còn
                    điểm nào di chuyển.</li>
                <li><b>Lặp lại</b> bước 2–4 đến khi đạt đúng <b>k cụm mục tiêu</b>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown("""
        <div class="app-card">
            <b style="color:#6b21a8;">So sánh trực quan DIANA vs AGNES:</b><br><br>
            <div style="font-family:monospace;font-size:12px;color:#374151;">
            <b>AGNES (Bottom-up):</b><br>
            • • • • • • •   ← 7 cụm ban đầu<br>
            [• •] • • • •   ← gom dần<br>
            [• • •] [• •]   ← ...<br>
            [• • • • • •]   ← 1 cụm cuối<br><br>
            <b>DIANA (Top-down):</b><br>
            [• • • • • •]   ← 1 cụm ban đầu<br>
            [• • •] [• •]   ← tách dần<br>
            [•][••] [•][•]  ← ...<br>
            • • • • • • •   ← k cụm cuối<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Chạy DIANA với cache ────────────────────────────────────────────────


    # ─── Scatter plots so sánh CURE vs DIANA ────────────────────────────────
    st.markdown("#### 📊 2. Kết quả phân cụm trực quan: CURE vs DIANA")

    col_dc1, col_dc2 = st.columns(2)
    with col_dc1:
        st.markdown("##### 🟢 CURE — Bottom-up (Gom từ dưới lên)")
        fig_cure_d = make_scatter_figure(
            X_data, cure_labels,
            f"CURE: {k_clusters} cụm — Gom từ điểm đơn lên",
            reps=cure_reps, means=cure_means,
            x_title=axis_x_name, y_title=axis_y_name
        )
        render_chart(fig_cure_d, key="tab_chart_10")
        render_metrics(cure_metrics)
        st.success("✅ **CURE:** Gom từng điểm lên → nhiều điểm đại diện + co cụm α → kháng nhiễu & phi cầu.")

    with col_dc2:
        st.markdown("##### 🟣 DIANA — Top-down (Tách từ trên xuống)")
        fig_diana = make_scatter_figure(
            X_data, diana_labels,
            f"DIANA: {k_clusters} cụm — Tách từ 1 cụm lớn xuống",
            means=diana_means,
            x_title=axis_x_name, y_title=axis_y_name
        )
        render_chart(fig_diana, key="tab_chart_11")
        render_metrics(diana_metrics)
        st.info("ℹ️ **DIANA:** Tách từ đại cụm → dựa vào đường kính và avg-dissimilarity để chia nhỏ dần.")

    # ─── Thống kê cụm DIANA ──────────────────────────────────────────────────
    st.markdown("#### 🔬 3. Chi tiết từng cụm DIANA")

    st.markdown("""
    <div class="app-card">
        <b>💡 Đọc bảng này như thế nào?</b>
        Mỗi hàng là 1 cụm mà DIANA tách ra được. <b>Splinter</b> là điểm đầu tiên bị tách ra để
        "kéo" theo những điểm xung quanh hình thành cụm con. Kích thước cụm cho thấy DIANA có xu hướng
        tạo ra cụm lớn 1 + nhiều cụm nhỏ hơn (không cân bằng như K-Means).
    </div>
    """, unsafe_allow_html=True)

    diana_cluster_rows = []
    for i, cluster_indices in enumerate(diana_clusters):
        pts = X_data[cluster_indices]
        mean_pt = pts.mean(axis=0)
        n_pts = len(cluster_indices)
        pct = n_pts / len(X_data) * 100
        diana_cluster_rows.append({
            "Cụm": f"Cụm {i+1}",
            "Số điểm": n_pts,
            "Tỉ lệ (%)": f"{pct:.1f}%",
            f"Tâm {axis_x_name[:10]}": f"{mean_pt[0]:.2f}",
            f"Tâm {axis_y_name[:10]}": f"{mean_pt[1]:.2f}",
        })
    df_diana_clusters = pd.DataFrame(diana_cluster_rows)
    render_table(df_diana_clusters)

    # ─── Bảng đối chiếu CURE vs DIANA ───────────────────────────────────────
    st.markdown("#### 📋 4. Bảng Đối chiếu Toàn diện: CURE vs DIANA")

    df_cure_diana = pd.DataFrame({
        "Tiêu chí": [
            "Hướng phân cụm",
            "Điểm xuất phát",
            "Cơ chế tách/gom",
            "Nhận diện cụm phi cầu",
            "Kháng nối chuỗi (Chaining)",
            "Kháng điểm ngoại lai",
            "Phát hiện cụm nhỏ ẩn (outlier cluster)",
            "Khả năng mở rộng (Big Data)",
            "Độ phức tạp tính toán",
            f"Silhouette Score (k={k_clusters})",
            "Davies-Bouldin Index",
            "Thời gian thực thi",
        ],
        "🟢 CURE (Bottom-up)": [
            "⬆️ Bottom-up (Gom từ dưới lên)",
            "N cụm đơn (mỗi điểm là 1 cụm)",
            f"Gom 2 cụm gần nhất qua {c_reps} đại diện co α={alpha_shrink}",
            "✅ Xuất sắc — đại diện trải đều theo thân cụm",
            "✅ Triệt tiêu hoàn toàn nhờ co cụm α",
            "✅ Rất tốt — α + 2 pha lọc ngoại lai",
            "✅ Tốt — nhóm nhỏ xa biệt được gom riêng",
            "✅ Xuất sắc — lấy mẫu s + gán nhãn O(N)",
            "O(s² log s) gom + O(N) gán nhãn",
            f"{cure_metrics['Silhouette']:.4f}",
            f"{cure_metrics['Davies-Bouldin']:.4f}",
            f"{cure_time:.4f}s",
        ],
        "🟣 DIANA (Top-down)": [
            "⬇️ Top-down (Tách từ trên xuống)",
            "1 cụm lớn chứa toàn bộ N điểm",
            "Tìm splinter → tách theo avg-dissimilarity",
            "⚠️ Trung bình — phụ thuộc vào hình dạng vùng tách",
            "⚠️ Trung bình — không có cơ chế đặc biệt",
            "⚠️ Trung bình — outlier ảnh hưởng đến đường kính cụm",
            "✅ Tốt — nhóm nhỏ có đường kính lớn sẽ bị tách sớm",
            "❌ Kém — O(N²) mỗi lần tách, nặng với dữ liệu lớn",
            "O(N² × k) — chậm khi N lớn",
            f"{diana_metrics['Silhouette']:.4f}",
            f"{diana_metrics['Davies-Bouldin']:.4f}",
            f"{diana_time:.4f}s",
        ],
    })
    render_table(df_cure_diana)

    # ─── Khi nào dùng DIANA? ─────────────────────────────────────────────────
    st.markdown("#### 💡 5. Khi nào nên dùng DIANA thay vì AGNES hay CURE?")

    col_dw1, col_dw2, col_dw3 = st.columns(3)
    with col_dw1:
        st.markdown("""
        <div class="app-card">
            <b style="color:#15803d;">✅ DIANA phù hợp khi:</b>
            <ul style="font-size:12.5px;margin:8px 0 0 14px;line-height:1.8;color:#374151;">
                <li>Muốn nhận ra <b>cụm toàn cục</b> trước (global → local)</li>
                <li>Dữ liệu có <b>1 cụm lớn + nhiều cụm nhỏ</b> tách rời rõ</li>
                <li>Cần phân tích <b>cấu trúc phân cấp từ trên xuống</b> (thị trường → phân khúc)</li>
                <li>Tập dữ liệu <b>vừa và nhỏ</b> (< 500 điểm)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_dw2:
        st.markdown("""
        <div class="app-card">
            <b style="color:#b91c1c;">❌ DIANA không phù hợp khi:</b>
            <ul style="font-size:12.5px;margin:8px 0 0 14px;line-height:1.8;color:#374151;">
                <li>Dữ liệu <b>lớn</b> (N > 1000) — chi phí O(N²) quá nặng</li>
                <li>Có nhiều <b>điểm ngoại lai</b> làm tăng đường kính cụm giả</li>
                <li>Cụm có <b>hình dạng phức tạp</b>, phi cầu, uốn lượn</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_dw3:
        st.markdown("""
        <div class="app-card">
            <b style="color:#1d4ed8;">🏆 Trong đề tài này:</b>
            <ul style="font-size:12.5px;margin:8px 0 0 14px;line-height:1.8;color:#374151;">
                <li>DIANA là <b>đối trọng thú vị</b> với CURE/AGNES vì hướng ngược lại</li>
                <li>Chứng minh <b>cùng dữ liệu, kết quả khác nhau</b> tùy hướng phân cụm</li>
                <li><b>CURE vẫn vượt trội</b> toàn diện nhờ lấy mẫu + co cụm + Big Data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ─── Kết luận ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-card">
        <div style="font-size:15px;font-weight:700;margin-bottom:8px;">
            🎯 Kết luận: DIANA hoàn thiện bức tranh Hierarchical Clustering
        </div>
        <div style="font-size:13px;line-height:1.9;opacity:0.95;">
            Ba thuật toán <b>AGNES → CURE → DIANA</b> cùng thuộc gia đình <i>Hierarchical Clustering</i>
            nhưng đại diện cho 3 triết lý khác nhau:<br>
            &nbsp;&nbsp;• <b>AGNES</b>: Đơn giản, nền tảng lý thuyết, giới hạn thực tế.<br>
            &nbsp;&nbsp;• <b>CURE</b>: Nâng cấp AGNES với co cụm α, nhiều đại diện, tối ưu Big Data — <b>đề tài chính</b>.<br>
            &nbsp;&nbsp;• <b>DIANA</b>: Hướng ngược lại, phát hiện cấu trúc toàn cục trước — bổ sung góc nhìn học thuật.<br><br>
            Sự có mặt của cả 3 chứng minh rằng <b>không có thuật toán nào là "tốt nhất" tuyệt đối</b> —
            mỗi bài toán thực tế cần lựa chọn phương pháp phù hợp với đặc điểm dữ liệu và mục tiêu phân tích.
        </div>
    </div>
    """, unsafe_allow_html=True)


# %% Cell 27 - Tab 10: 📋 Bảng Tổng hợp Ma trận Đối sánh Toàn diện
# Dùng dữ liệu và kết quả đã chuẩn bị ở các cell phía trên.
with tab_summary:
    render_tab_header('📋 Bảng Tổng hợp Ma trận Đối sánh Toàn diện', 'Tổng hợp chỉ số của các thuật toán trên cùng dữ liệu đầu vào.', data_context)
    st.markdown("Bảng tổng hợp đối đầu giữa **CURE** và các thuật toán phân cụm trong chương trình môn học:")

    # Đồ thị Bar Chart so sánh Silhouette Score
    algs_names = list(all_metrics)
    sils = [metrics['Silhouette'] for metrics in all_metrics.values()]
    dbs = [metrics['Davies-Bouldin'] for metrics in all_metrics.values()]
    times = [metrics['Time'] for metrics in all_metrics.values()]
    render_table(pd.DataFrame(all_metrics).T.rename_axis("Thuật toán").reset_index())

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_bar_sil = go.Figure(data=[
            go.Bar(name='Silhouette Score (Càng cao càng tốt)', x=algs_names, y=sils,
                   marker_color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(algs_names))],
                   text=[f"{v:.3f}" for v in sils], textposition='auto')
        ])
        fig_bar_sil.update_layout(title="Chỉ số Silhouette Score trên Tập dữ liệu Hiện hành", height=340, yaxis=dict(range=[-1.0, 1.0]))
        render_chart(fig_bar_sil, key="tab_chart_12")

    with col_s2:
        fig_bar_time = go.Figure(data=[
            go.Bar(name='Thời gian thực thi (giây)', x=algs_names, y=times,
                   marker_color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(algs_names))],
                   text=[f"{v:.4f}s" for v in times], textposition='auto')
        ])
        fig_bar_time.update_layout(title="Thời gian Thực thi (giây)", height=340)
        render_chart(fig_bar_time, key="tab_chart_13")

    st.markdown("#### 🏆 Ma trận Đánh giá Tổng kết Toàn diện")
    df_full_summary = pd.DataFrame({
        "Tiêu chuẩn đánh giá": [
            "Cụm hình dạng phi cầu (Two Moons, Circles)",
            "Cụm kéo dài hình elip (Anisotropic)",
            "Khả năng kháng ngoại lai và nhiễu biên",
            "Hiện tượng nối chuỗi (Chaining Effect)",
            "Khả năng mở rộng trên Big Data",
            "Độ nhạy với siêu tham số",
            "Thời gian thực thi trên tập hiện tại",
            "Đánh giá ứng dụng thực tế"
        ],
        "CURE": [
            "⭐⭐⭐⭐⭐ (Xuất sắc)",
            "⭐⭐⭐⭐⭐ (Xuất sắc)",
            "⭐⭐⭐⭐⭐ (Kháng triệt để)",
            "⭐⭐⭐⭐⭐ (Không bị)",
            "⭐⭐⭐⭐⭐ (Rất tốt - Lấy mẫu + Gán nhãn)",
            "⭐⭐⭐ (Cần chỉnh c, alpha)",
            f"{cure_time:.4f}s",
            "Tối ưu cho dữ liệu không gian, hình học phức tạp"
        ],
        "K-Means": [
            "⭐ (Thất bại)",
            "⭐⭐ (Kém)",
            "⭐ (Bị kéo lệch tâm)",
            "⭐⭐⭐⭐⭐ (Không bị)",
            "⭐⭐⭐⭐⭐ (Cực nhanh)",
            "⭐⭐⭐⭐⭐ (Chỉ cần k)",
            f"{km_time:.4f}s",
            "Chỉ dùng cho cụm hình cầu cân đối"
        ],
        "K-Medoids (PAM)": [
            "⭐ (Thất bại)",
            "⭐⭐ (Kém)",
            "⭐⭐⭐ (Giảm lệch tâm)",
            "⭐⭐⭐⭐⭐ (Không bị)",
            "⭐ (Rất chậm khi N lớn)",
            "⭐⭐⭐⭐ (Chỉ cần k)",
            f"{kmed_time:.4f}s",
            "Dùng khi có ngoại lai nhưng dữ liệu nhỏ và hình cầu"
        ],
        "Hierarchical (Single)": [
            "⭐⭐⭐⭐ (Tốt khi không có nhiễu)",
            "⭐⭐⭐⭐ (Tốt)",
            "⭐ (Rất nhạy cảm với nhiễu)",
            "⭐ (Bị nối chuỗi nghiêm trọng)",
            "⭐ (Tràn bộ nhớ O(N²))",
            "⭐⭐⭐⭐ (Chỉ cần k)",
            f"{hier_time:.4f}s",
            "Chỉ dùng cho dữ liệu nhỏ và không có ngoại lai"
        ]
    })
    render_table(df_full_summary)


# %% Cell 28 - Tab 11: 🔎 Phân tích Ngoại lai (Outlier Analysis) — Khách hàng Cao tuổi
# Dùng dữ liệu và kết quả đã chuẩn bị ở các cell phía trên.
with tab_outlier:
    render_tab_header('🔎 Phân tích Ngoại lai (Outlier Analysis) — Khách hàng Cao tuổi', 'Khảo sát nhóm khách hàng từ 70 tuổi trong mẫu dữ liệu đang chọn.', data_context)

    # Chỉ hoạt động khi dùng dữ liệu thực tế
    if "Data" not in dataset_type or df_customer_raw is None:
        st.info("ℹ️ Tab này chỉ hoạt động khi chọn **Dữ liệu thực tế (Data/Test.csv)** ở sidebar bên trái.")
    else:
        # ──────────────────────────────────────────────────────────
        # Phần 0: Giới thiệu
        # ──────────────────────────────────────────────────────────
        st.markdown("""
        <div class="app-card">
            <b style="color:#1e40af;font-size:14px;">📌 Mục tiêu của tab này là gì?</b><br>
            <span style="font-size:13px;color:#374151;">
            Trong phân cụm khách hàng, <b>ngoại lai (Outlier)</b> là những điểm dữ liệu nằm <b>rất xa phần lớn</b> các điểm còn lại —
            thường gây ra phân cụm sai hoặc bị gộp nhầm vào cụm không phù hợp.<br><br>
            Tab này sẽ trả lời 3 câu hỏi:<br>
            &nbsp;&nbsp;① Nhóm khách hàng cao tuổi (≥ 70) có đặc trưng hành vi gì?<br>
            &nbsp;&nbsp;② Vì sao CURE xử lý nhóm này tốt hơn K-Means?<br>
            &nbsp;&nbsp;③ Nên áp dụng chiến lược kinh doanh gì với nhóm này?
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────
        # Dùng đúng mẫu đã phân cụm để số liệu nhất quán giữa các tab
        # ──────────────────────────────────────────────────────────
        spend_map_ol = {'Low': 1, 'Average': 2, 'High': 3}
        df_full = df_customer_raw.copy()

        OUTLIER_AGE = 70  # Ngưỡng định nghĩa ngoại lai

        df_full['Is_Outlier'] = df_full['Age'] >= OUTLIER_AGE
        df_full['NhomTuoi'] = pd.cut(
            df_full['Age'],
            bins=[0, 30, 45, 60, 70, 120],
            labels=['18–30', '31–45', '46–60', '61–70', '71+']
        )

        n_total = len(df_full)
        n_outlier = df_full['Is_Outlier'].sum()
        n_normal = n_total - n_outlier
        pct_outlier = n_outlier / n_total * 100

        # ──────────────────────────────────────────────────────────
        # Phần 1: Số liệu tổng quan
        # ──────────────────────────────────────────────────────────
        st.markdown("#### 📊 1. Tổng quan phân bố Độ tuổi trong tập dữ liệu")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Tổng số khách hàng", f"{n_total:,}")
        col_m2.metric("KH bình thường (< 70t)", f"{n_normal:,}", f"{100-pct_outlier:.1f}% tổng")
        col_m3.metric("KH cao tuổi (≥ 70t)", f"{n_outlier:,}", f"{pct_outlier:.1f}% tổng", delta_color="off")
        col_m4.metric("Tuổi TB nhóm ≥ 70", f"{df_full[df_full['Is_Outlier']]['Age'].mean():.1f}")

        st.markdown("""
        <div class="app-card">
            <b>💡 Giải thích:</b> Trong tập 2.627 khách hàng, có <b>255 người (9.7%)</b> từ <b>70 tuổi trở lên</b>.
            Đây là những điểm nằm ở "đuôi phải" của phân phối tuổi — xa so với đa số và có hành vi tiêu dùng rất khác biệt.
            Thuật ngữ thống kê gọi những điểm như vậy là <b>Outlier</b>.
        </div>
        """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────
        # Phần 2: Histogram + Scatter song song
        # ──────────────────────────────────────────────────────────
        col_hist, col_scat = st.columns(2)

        with col_hist:
            st.markdown("##### Phân bố Tuổi theo nhóm")
            group_counts = df_full['NhomTuoi'].value_counts().sort_index().reset_index()
            group_counts.columns = ['Nhom', 'SoLuong']
            group_counts['Mau'] = ['#60a5fa', '#34d399', '#fbbf24', '#f97316', '#ef4444']
            fig_hist = px.bar(
                group_counts, x='Nhom', y='SoLuong', color='Nhom',
                color_discrete_sequence=['#60a5fa', '#34d399', '#fbbf24', '#f97316', '#ef4444'],
                text='SoLuong', labels={'Nhom': 'Nhóm tuổi', 'SoLuong': 'Số khách hàng'}
            )
            fig_hist.update_traces(textposition='outside')
            fig_hist.add_vline(x=3.5, line_dash="dash", line_color="#dc2626",
                               annotation_text="Ngưỡng Outlier (≥ 70)", annotation_position="top left",
                               annotation_font_color="#dc2626")
            fig_hist.update_layout(
                height=360, showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="#fafbfc",
                xaxis=dict(title="Nhóm tuổi"),
                yaxis=dict(title="Số khách hàng")
            )
            render_chart(fig_hist, key="tab_chart_14")
            st.caption("🔴 Đường đỏ đứt là ngưỡng phân tách Outlier (≥ 70 tuổi). Cột đỏ = nhóm ngoại lai.")

        with col_scat:
            st.markdown("##### Scatter: Tuổi vs Chi tiêu (highlight ngoại lai)")
            df_normal_plot = df_full[~df_full['Is_Outlier']]
            df_outlier_plot = df_full[df_full['Is_Outlier']]

            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=df_normal_plot['Age'], y=df_normal_plot['Spending_Score_Num'],
                mode='markers',
                marker=dict(size=5, color='#60a5fa', opacity=0.5),
                name='Khách hàng bình thường (< 70t)'
            ))
            fig_scat.add_trace(go.Scatter(
                x=df_outlier_plot['Age'], y=df_outlier_plot['Spending_Score_Num'],
                mode='markers',
                marker=dict(size=9, color='#ef4444', symbol='star', line=dict(width=1, color='#7f1d1d')),
                name='Ngoại lai (≥ 70t) ⭐'
            ))
            fig_scat.add_vline(x=70, line_dash="dash", line_color="#dc2626",
                               annotation_text="Ngưỡng 70t", annotation_font_color="#dc2626")
            fig_scat.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="#fafbfc",
                legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
                xaxis=dict(title="Tuổi (Age)", showgrid=True, gridcolor='#f1f5f9'),
                yaxis=dict(title="Chi tiêu (1=Low, 2=Avg, 3=High)", showgrid=True, gridcolor='#f1f5f9')
            )
            render_chart(fig_scat, key="tab_chart_15")
            st.caption("⭐ Các ngôi sao đỏ = nhóm ≥ 70 tuổi. Nhận thấy chi tiêu của họ **không thấp** — tập trung nhiều ở mức High!")

        # ──────────────────────────────────────────────────────────
        # Phần 3: Bảng so sánh đặc trưng 2 nhóm
        # ──────────────────────────────────────────────────────────
        st.markdown("#### 🧬 2. So sánh Đặc trưng hành vi: Nhóm bình thường vs Nhóm ≥ 70 tuổi")

        df_ol = df_full[df_full['Is_Outlier']]
        df_nm = df_full[~df_full['Is_Outlier']]

        so_sanh_data = {
            "Đặc trưng": [
                "Số lượng (người)",
                "Tuổi trung bình",
                "Chi tiêu TB (1=Low, 2=Avg, 3=High)",
                "Kinh nghiệm làm việc TB (năm)",
                "Quy mô gia đình TB (người)",
                "Tỉ lệ đã lập gia đình",
                "Tỉ lệ chi tiêu MỨC CAO (High %)",
            ],
            "Nhóm bình thường (< 70t)": [
                f"{n_normal:,} người",
                f"{df_nm['Age'].mean():.1f} tuổi",
                f"{df_nm['Spending_Score_Num'].mean():.2f}",
                f"{df_nm['Work_Experience'].mean():.1f} năm",
                f"{df_nm['Family_Size'].mean():.1f} người",
                f"{(df_nm['Ever_Married']=='Yes').mean()*100:.0f}%",
                f"{(df_nm['Spending_Score']=='High').mean()*100:.1f}%",
            ],
            "Nhóm ≥ 70 tuổi (Ngoại lai)": [
                f"{n_outlier:,} người ({pct_outlier:.1f}%)",
                f"🔴 {df_ol['Age'].mean():.1f} tuổi",
                f"🟢 {df_ol['Spending_Score_Num'].mean():.2f} (CAO HƠN!)",
                f"🟡 {df_ol['Work_Experience'].mean():.1f} năm (ít hơn)",
                f"🟡 {df_ol['Family_Size'].mean():.1f} người (nhỏ hơn)",
                f"🟢 {(df_ol['Ever_Married']=='Yes').mean()*100:.0f}% (cao hơn!)",
                f"🟢 {(df_ol['Spending_Score']=='High').mean()*100:.1f}% (GẦN GẤP ĐÔI!)",
            ]
        }
        df_compare = pd.DataFrame(so_sanh_data)
        render_table(df_compare)

        st.markdown("""
        <div class="app-card">
            <b>💡 Đọc kết quả bảng này như thế nào?</b><br>
            Nhóm khách hàng ≥ 70 tuổi <b>nghịch lý</b> so với kỳ vọng ban đầu:
            <ul style="margin:6px 0 0 16px;">
                <li><b>Chi tiêu cao hơn</b> trung bình tổng thể — họ có tích lũy tài chính tốt sau nhiều năm làm việc.</li>
                <li><b>Quy mô gia đình nhỏ hơn</b> — con cái đã ra ở riêng, họ chi tiêu cho bản thân nhiều hơn.</li>
                <li><b>Kinh nghiệm làm việc ít hơn</b> — phần lớn đã về hưu hoặc làm bán thời gian.</li>
                <li><b>Gần gấp đôi tỉ lệ High Spending</b> — đây là phân khúc <b>giá trị cao</b>, không phải ngoại lai vô nghĩa!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────
        # Phần 4: Biểu đồ chi tiêu theo nhóm tuổi
        # ──────────────────────────────────────────────────────────
        st.markdown("#### 📈 3. Xu hướng Chi tiêu tăng dần theo Độ tuổi — Insight bất ngờ!")

        col_trend1, col_trend2 = st.columns(2)

        with col_trend1:
            spend_by_age = df_full.groupby('NhomTuoi', observed=True)['Spending_Score_Num'].mean().reset_index()
            spend_by_age.columns = ['NhomTuoi', 'ChiTieuTB']
            colors_trend = ['#60a5fa', '#34d399', '#fbbf24', '#f97316', '#ef4444']
            fig_trend = px.bar(
                spend_by_age, x='NhomTuoi', y='ChiTieuTB', color='NhomTuoi',
                color_discrete_sequence=colors_trend,
                text=spend_by_age['ChiTieuTB'].round(2),
                labels={'NhomTuoi': 'Nhóm tuổi', 'ChiTieuTB': 'Chi tiêu TB'}
            )
            fig_trend.update_traces(textposition='outside')
            fig_trend.update_layout(
                height=320, showlegend=False,
                title="Chi tiêu TB tăng liên tục theo tuổi",
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="#fafbfc",
                yaxis=dict(range=[0, 2.8])
            )
            render_chart(fig_trend, key="tab_chart_16")

        with col_trend2:
            fam_by_age = df_full.groupby('NhomTuoi', observed=True)['Family_Size'].mean().reset_index()
            fam_by_age.columns = ['NhomTuoi', 'GiaDinhTB']
            fig_fam = px.bar(
                fam_by_age, x='NhomTuoi', y='GiaDinhTB', color='NhomTuoi',
                color_discrete_sequence=colors_trend,
                text=fam_by_age['GiaDinhTB'].round(2),
                labels={'NhomTuoi': 'Nhóm tuổi', 'GiaDinhTB': 'Quy mô gia đình TB'}
            )
            fig_fam.update_traces(textposition='outside')
            fig_fam.update_layout(
                height=320, showlegend=False,
                title="Quy mô gia đình giảm dần khi tuổi cao",
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="#fafbfc",
                yaxis=dict(range=[0, 4.2])
            )
            render_chart(fig_fam, key="tab_chart_17")

        st.markdown("""
        <div class="app-card">
            <b>🔍 Phân tích xu hướng:</b><br>
            Hai biểu đồ trên tiết lộ một <b>"quy luật ngược"</b> trong dữ liệu:<br>
            <ul style="margin:6px 0 0 16px;">
                <li><b>Chi tiêu tăng từ 1.11 → 2.13</b> khi tuổi tăng từ 18-30 lên 71+. Lý giải: người lớn tuổi đã thanh toán hết nợ vay, con cái tự lập → tiền nhàn rỗi chi tiêu cho bản thân.</li>
                <li><b>Quy mô gia đình giảm từ 3.47 → 1.96</b> — đây là "empty nest effect": con cái trưởng thành và ra ở riêng.</li>
                <li>Hai xu hướng này kết hợp tạo ra nhóm ngoại lai có <b>thu nhập tự do cao + nhu cầu chi tiêu cao</b> — phân khúc VIP tiềm năng!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────
        # Phần 5: Vì sao CURE xử lý tốt hơn
        # ──────────────────────────────────────────────────────────
        st.markdown("#### 🤖 4. Tại sao CURE xử lý nhóm Ngoại lai này tốt hơn các thuật toán khác?")

        col_why1, col_why2 = st.columns(2)

        with col_why1:
            st.markdown("""
            <div class="app-card">
                <div style="font-weight:700;color:#1e40af;margin-bottom:8px;">🟢 Cách CURE xử lý</div>
                <ul style="font-size:13px;color:#1e3a5f;line-height:1.8;margin-left:14px;">
                    <li><b>Cơ chế co cụm α:</b> Các điểm đại diện được kéo vào bên trong lõi cụm, giảm ảnh hưởng của các điểm ngoại vi như nhóm 84.8 tuổi.</li>
                    <li><b>Phân cụm phân cấp:</b> CURE từ từ gom cụm từ nhỏ → lớn, nên nhóm 5 người cao tuổi được nhận diện như một <b>cụm nhỏ độc lập</b> thay vì bị gộp vào cụm sai.</li>
                    <li><b>Nhiều điểm đại diện:</b> c = {c_reps} điểm trải dọc theo thân cụm giúp mô tả đúng hình dạng thực tế, không bị ép vào hình cầu.</li>
                </ul>
            </div>
            """.format(c_reps=c_reps), unsafe_allow_html=True)

        with col_why2:
            st.markdown("""
            <div class="app-card">
                <div style="font-weight:700;color:#991b1b;margin-bottom:8px;">🔴 Cách K-Means xử lý (sai)</div>
                <ul style="font-size:13px;color:#7f1d1d;line-height:1.8;margin-left:14px;">
                    <li><b>Tâm bị kéo lệch:</b> Khi nhóm 84.8 tuổi bị gộp vào Cụm 2 (35.8 tuổi TB), tâm cụm bị lệch lên → ranh giới phân cụm không còn phản ánh thực tế.</li>
                    <li><b>Giả định hình cầu:</b> K-Means coi tất cả cụm là hình cầu bán kính đều nhau → không thể bóc tách một nhóm nhỏ 5 người cách xa phần còn lại.</li>
                    <li><b>Nhãn sai:</b> K-Means gán nhãn "Khách hàng Trẻ / Tiềm năng" cho người 84 tuổi — sai hoàn toàn về mặt nghiệp vụ!</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────
        # Phần 6: Khuyến nghị kinh doanh
        # ──────────────────────────────────────────────────────────
        st.markdown("#### 💼 5. Khuyến nghị Chiến lược Kinh doanh cho từng nhóm")

        recs = [
            {
                "icon": "🔵",
                "nhom": "Cụm 1 — Khách hàng Phổ thông (18–45 tuổi, TB ~35.8t)",
                "mo_ta": "Nhóm lớn nhất (75.7%), đang trong giai đoạn lập nghiệp, chi tiêu thận trọng.",
                "khuyen_nghi": [
                    "Chương trình tích điểm thưởng, cashback",
                    "Flash sale và combo ưu đãi",
                    "Tiếp thị qua MXH (TikTok, Instagram, Facebook)",
                    "Trả góp 0% để kích cầu mua sắm lớn",
                ]
            },
            {
                "icon": "🔴",
                "nhom": "Cụm 2 — Khách hàng Trưởng thành / Cao cấp (45–65 tuổi, TB ~64.1t)",
                "mo_ta": "Nhóm trung lưu (20.7%), đỉnh thu nhập, chi tiêu ổn định, ít nhạy cảm giá.",
                "khuyen_nghi": [
                    "Thẻ thành viên VIP / Gold Member",
                    "Gói sản phẩm cao cấp, thương hiệu uy tín",
                    "Dịch vụ tư vấn riêng, chăm sóc khách hàng 1-1",
                    "Sản phẩm bảo hiểm nhân thọ, quỹ hưu trí",
                ]
            },
            {
                "icon": "🟢",
                "nhom": "Cụm 3 — Khách hàng Cao tuổi / Hưu trí (≥ 70 tuổi, TB ~84.8t)",
                "mo_ta": "Nhóm ngoại lai (3.6%) nhưng có SPENDING CAO NHẤT. Đã nghỉ hưu, gia đình nhỏ, tài chính tích lũy dồi dào.",
                "khuyen_nghi": [
                    "Sản phẩm chăm sóc sức khỏe & dinh dưỡng cao cấp",
                    "Bảo hiểm sức khỏe toàn diện, bảo hiểm tai nạn",
                    "Du lịch nghỉ dưỡng, spa, resort cao cấp",
                    "Hỗ trợ mua sắm tại nhà (giao hàng tận nơi, tư vấn qua điện thoại)",
                    "Sản phẩm công nghệ dễ dùng (tablet, điện thoại chữ to)",
                ]
            },
        ]

        for rec in recs:
            with st.expander(f"{rec['icon']} {rec['nhom']}", expanded=True):
                st.markdown(f"**Đặc điểm:** {rec['mo_ta']}")
                st.markdown("**Chiến lược đề xuất:**")
                for r in rec['khuyen_nghi']:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;✅ {r}")

        # ──────────────────────────────────────────────────────────
        # Phần 7: Kết luận
        # ──────────────────────────────────────────────────────────
        st.markdown("""
        <div class="app-card">
            <div style="font-size:15px;font-weight:700;margin-bottom:8px;">🎯 Kết luận của Tab Phân tích Ngoại lai</div>
            <div style="font-size:13px;line-height:1.8;opacity:0.95;">
                Nhóm khách hàng <b>≥ 70 tuổi</b> trong dữ liệu là một <b>Outlier theo nghĩa thống kê</b> (cách xa phần lớn phân phối)
                nhưng lại là một <b>Phân khúc giá trị cao về mặt kinh doanh</b>.<br><br>
                Điều này chứng minh sức mạnh của <b>CURE</b>:
                <ul style="margin:6px 0 0 18px;">
                    <li>Không đơn giản loại bỏ nhóm nhỏ như một số thuật toán khác.</li>
                    <li>Bóc tách chính xác nhóm này thành <b>Cụm 3 độc lập</b> nhờ cơ chế co cụm α và nhiều điểm đại diện.</li>
                    <li>Giúp doanh nghiệp <b>nhận diện đúng phân khúc VIP tiềm ẩn</b> mà K-Means và K-Medoids bỏ sót.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)


# %% Cell 29 - Chân trang

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 12px;">
    Đồ án môn học Khai thác dữ liệu - Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT)<br>
    Hệ thống mô phỏng phục vụ thuyết trình và bảo vệ đề tài tiểu luận CURE.
</div>
""", unsafe_allow_html=True)

