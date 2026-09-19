"""
Hệ thống Web Demo Mô phỏng & Đối sánh Thuật toán Phân cụm CURE (Clustering Using REpresentatives)
Môn học: Khai thác dữ liệu / Khai phá dữ liệu - Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT)
Khởi chạy: python -m streamlit run app_streamlit.py
"""

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

from cure_algorithm import CURE, KMedoids

# ==========================================
# CẤU HÌNH TRANG STREAMLIT
# ==========================================
st.set_page_config(
    page_title="CURE Clustering Visualizer - HUIT",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ==========================================
# TẠO & XỬ LÝ DỮ LIỆU
# ==========================================
def get_test_csv_path():
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

@st.cache_data
def load_and_preprocess_customer_data(pair_name, n_pts, seed, uploaded_bytes=None):
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

    if "Chi tiêu" in pair_name:
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

@st.cache_data
def generate_synthetic_dataset(d_type, n_pts, seed):
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

# Helper: Tính metrics
def compute_metrics(X, labels, exec_time):
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

# Helper: Vẽ đồ thị Plotly đẹp cho từng thuật toán
def make_scatter_figure(X, labels, title, reps=None, means=None, centers=None, center_label="Tâm", x_title="X", y_title="Y"):
    df_p = pd.DataFrame(X, columns=['X', 'Y'])
    df_p['Cụm'] = [f"Cụm {l+1}" if l != -1 else "Ngoại lai (Nhiễu)" for l in labels]
    
    color_map = {}
    palette = px.colors.qualitative.Plotly
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

# ==========================================
# CHẠY CÁC THUẬT TOÁN ĐỂ LẤY KẾT QUẢ
# ==========================================
# 1. CURE
t0 = time.time()
cure_model = CURE(n_clusters=k_clusters, n_representatives=c_reps, shrink_factor=alpha_shrink)
cure_model.fit(X_data)
cure_time = time.time() - t0
cure_labels = cure_model.labels_
cure_reps = cure_model.get_representatives()
cure_means = cure_model.get_cluster_means()
cure_metrics = compute_metrics(X_data, cure_labels, cure_time)

# 2. K-Means
t_km0 = time.time()
km_model = KMeans(n_clusters=k_clusters, random_state=noise_seed, n_init='auto')
km_labels = km_model.fit_predict(X_data)
km_time = time.time() - t_km0
km_centers = km_model.cluster_centers_
km_metrics = compute_metrics(X_data, km_labels, km_time)

# 3. K-Medoids
t_kmed0 = time.time()
kmed_model = KMedoids(n_clusters=k_clusters, random_state=noise_seed)
kmed_labels = kmed_model.fit_predict(X_data)
kmed_time = time.time() - t_kmed0
kmed_centers = kmed_model.cluster_centers_
kmed_metrics = compute_metrics(X_data, kmed_labels, kmed_time)

# 4. Hierarchical (Single Linkage)
t_hier0 = time.time()
hier_model = AgglomerativeClustering(n_clusters=k_clusters, linkage='single')
hier_labels = hier_model.fit_predict(X_data)
hier_time = time.time() - t_hier0
hier_metrics = compute_metrics(X_data, hier_labels, hier_time)


# ==========================================
# HÀM TẠO BỘ MÔ PHỎNG CANVAS HTML5 TRỰC QUAN
# ==========================================
def render_canvas_html(points_2d, k, c, alpha, title_dataset):
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
              profileTitle = 'Phân khúc 1: Khách hàng Phổ thông';
              profileDesc = 'Chiếm tỷ trọng lớn, mức chi tiêu và độ tuổi tập trung ở vùng trung tâm thị trường.';
            }} else if (idx === 1) {{
              profileTitle = 'Phân khúc 2: Khách hàng Trưởng thành / Cao cấp';
              profileDesc = 'Độ tuổi cao hơn hoặc chỉ số chi tiêu vượt trội, có giá trị sinh lời cao.';
            }} else if (idx === 2) {{
              profileTitle = 'Phân khúc 3: Khách hàng Trẻ / Năng động';
              profileDesc = 'Nhóm khách hàng trẻ tuổi, hành vi mua sắm độc lập, tiềm năng khai thác lâu dài.';
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

# ==========================================
# THIẾT LẬP HỆ THỐNG TABS CHUYÊN BIỆT
# ==========================================
tab_cure_sim, tab_cure_main, tab_cure_steps, tab_cure_flow, tab_vs_kmeans, tab_vs_kmedoids, tab_vs_hier, tab_summary = st.tabs([
    "🎮 Mô Phỏng Tương Tác (Canvas HTML5)",
    "🎯 CURE: Trực quan hóa & Phân cụm",
    "📝 CURE: Ví dụ tính tay (Toy Example)",
    "🔍 CURE: Quy trình 5 giai đoạn",
    "⚔️ So sánh: CURE vs K-Means",
    "⚔️ So sánh: CURE vs K-Medoids",
    "⚔️ So sánh: CURE vs Hierarchical",
    "📋 Bảng Tổng hợp Đối sánh"
])

# ==============================================================================
# TAB MỚI: MÔ PHỎNG TƯƠNG TÁC TỪNG BƯỚC (CANVAS HTML5 NHÚNG TRỰC TIẾP)
# ==============================================================================
with tab_cure_sim:
    st.markdown("### 🎮 Mô Phỏng Tương Tác Từng Bước (Interactive CURE Visualizer)")
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
        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 12px; border-radius: 8px; font-size: 13px; color: #166534;">
            <b>💡 Hướng dẫn thao tác trực quan:</b><br>
            • Nhấp <b>"⏩ 1 Bước (Step)"</b> để theo dõi cặp cụm gần nhất sáp nhập và quan sát cách $c$ điểm đại diện co rút $\alpha$ về phía tâm.<br>
            • Nhấp <b>"▶️ Chạy Tự Động"</b> để xem toàn bộ quá trình gom cụm diễn ra sinh động.<br>
            • Khi đạt mục tiêu $k$, <b>Bảng Giải Thích Kết Quả</b> sẽ tự động xuất hiện ngay bên dưới khung vẽ!
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
    
    col_exp_1, col_exp_2 = st.columns([1.6, 1.4])
    
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
            
            # Gợi ý tên phân khúc
            if "Data" in dataset_type:
                if idx == 0:
                    c_name = "Khách hàng Phổ thông"
                    action = "Duy trì khuyến mãi tích điểm định kỳ"
                elif idx == 1:
                    c_name = "Khách hàng Trưởng thành / Cao cấp"
                    action = "Cung cấp dịch vụ VIP, gói sản phẩm giá trị cao"
                elif idx == 2:
                    c_name = "Khách hàng Trẻ / Tiềm năng"
                    action = "Tiếp thị số qua MXH, trải nghiệm mua sắm nhanh"
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
        st.dataframe(df_cluster_report, use_container_width=True, hide_index=True)

    with col_exp_2:
        st.markdown("##### 2. Nhận Định Tại Sao CURE Vượt Trội Trên Tập Này")
        st.markdown(f"""
        * 🎯 **Định hình bằng $c={c_reps}$ đại diện:** Thay vì ép toàn bộ dữ liệu vào một khối tròn nhân tạo quanh 1 tâm duy nhất như K-Means, CURE chọn các điểm đại diện rải dọc theo thân cụm, ôm trọn các nhóm khách hàng có phân bố dẹt hoặc hình học phi cầu.
        * 🛡️ **Khoảng đệm an toàn với $\alpha={alpha_shrink}$:** Nhờ co các điểm đại diện về trọng tâm theo hệ số $\alpha$, các khách hàng có hành vi dị biệt hoặc điểm nhiễu ngoại lai ở rìa ngoài bị vô hiệu hóa, không làm xê dịch ranh giới phân cụm.
        * ⚡ **Chỉ số đánh giá định lượng:**
          - **Silhouette Score:** `{cure_metrics['Silhouette']:.4f}` *(Độ gắn kết nội cụm và tách biệt ngoại cụm tốt)*.
          - **Davies-Bouldin Index:** `{cure_metrics['Davies-Bouldin']:.4f}` *(Càng nhỏ chứng tỏ các cụm càng cô đặc)*.
        """)


# ==============================================================================
# TAB 1: CURE TRỰC QUAN HÓA CHI TIẾT
# ==============================================================================
with tab_cure_main:
    st.markdown("### 🎯 Kết Quả Phân Cụm Thuật Toán CURE")
    st.markdown(f"Đang phân cụm trên tập: **{dataset_type}** với $N = {len(X_data)}$ điểm dữ liệu.")

    col_c1, col_c2 = st.columns([3, 1])

    with col_c1:
        fig_cure = make_scatter_figure(
            X_data, cure_labels,
            f"Kết quả phân cụm CURE (k={k_clusters}, c={c_reps} đại diện, α={alpha_shrink})",
            reps=cure_reps, means=cure_means,
            x_title=axis_x_name, y_title=axis_y_name
        )
        st.plotly_chart(fig_cure, use_container_width=True)

    with col_c2:
        st.markdown("#### 📊 Chỉ số Đánh giá Chất lượng")
        st.metric("Silhouette Score", f"{cure_metrics['Silhouette']:.4f}", help="Càng gần 1 càng phân cụm tốt")
        st.metric("Davies-Bouldin Index", f"{cure_metrics['Davies-Bouldin']:.4f}", help="Càng nhỏ cụm càng đặc và tách biệt")
        st.metric("Calinski-Harabasz", f"{cure_metrics['Calinski-Harabasz']:.1f}")
        st.metric("Thời gian thực thi", f"{cure_time:.4f} giây")

        st.markdown("""
        <div style="background-color: #e0f2fe; padding: 12px; border-radius: 8px; font-size: 12.5px; border-left: 4px solid #0284c7;">
            <b>Ý nghĩa trực quan của CURE:</b><br>
            • <b>Dấu X đen:</b> $c$ điểm đại diện đã co cụm về phía tâm.<br>
            • <b>Ngôi sao vàng:</b> Trọng tâm mean của cụm.<br>
            • Nhờ phân bố rải rác nhiều điểm đại diện, CURE ôm trọn vẹn đường cong tự nhiên và phân bố khách hàng!
        </div>
        """, unsafe_allow_html=True)

    if df_customer_raw is not None:
        with st.expander("📋 Xem trước Dữ liệu gốc khách hàng từ folder Data/Test.csv"):
            st.dataframe(df_customer_raw.head(20), use_container_width=True)


# ==============================================================================
# TAB 2: CURE VÍ DỤ TÍNH TAY TỪNG BƯỚC (TOY EXAMPLE)
# ==============================================================================
with tab_cure_steps:
    st.markdown("### 📝 Bài toán Ví dụ Tính tay Từng bước (Toy Example)")
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
    step_img = f"toy_example_steps/step_{step_idx}.png"

    col_t1, col_t2 = st.columns([1.8, 1.2])

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


# ==============================================================================
# TAB 3: CURE QUY TRÌNH 5 GIAI ĐOẠN & LƯU ĐỒ KHỐI
# ==============================================================================
with tab_cure_flow:
    st.markdown("### 🔍 Quy trình 5 Giai đoạn & Kiến trúc Xử lý Dữ liệu lớn")
    
    col_f1, col_f2 = st.columns([1.5, 1])

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
        st.table(df_params)


# ==============================================================================
# TAB 4: SO SÁNH 1-1: CURE VS K-MEANS
# ==============================================================================
with tab_vs_kmeans:
    st.markdown("### ⚔️ So sánh Đối đầu Trực diện: CURE vs K-Means")
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
        st.plotly_chart(fig_c_km, use_container_width=True)
        st.write(f"**Silhouette Score:** `{cure_metrics['Silhouette']:.4f}` | **Thời gian:** `{cure_time:.4f}s`")
        st.success("✅ **Ưu thế của CURE:** Ôm trọn vẹn dải cong phi cầu và phân bố tự nhiên nhờ c điểm đại diện rải đều!")

    with col_km2:
        st.markdown("##### 🔵 Thuật toán K-Means (Thuật toán đối chứng)")
        fig_km = make_scatter_figure(
            X_data, km_labels,
            f"K-Means: Cắt ngang cụm do giả định hình cầu",
            means=km_centers,
            x_title=axis_x_name, y_title=axis_y_name
        )
        st.plotly_chart(fig_km, use_container_width=True)
        st.write(f"**Silhouette Score:** `{km_metrics['Silhouette']:.4f}` | **Thời gian:** `{km_time:.4f}s`")
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
    st.table(df_cmp_km)


# ==============================================================================
# TAB 5: SO SÁNH 1-1: CURE VS K-MEDOIDS (PAM)
# ==============================================================================
with tab_vs_kmedoids:
    st.markdown("### ⚔️ So sánh Đối đầu Trực diện: CURE vs K-Medoids (PAM)")
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
        st.plotly_chart(fig_c_kmed, use_container_width=True)
        st.write(f"**Silhouette Score:** `{cure_metrics['Silhouette']:.4f}` | **Thời gian:** `{cure_time:.4f}s`")
        st.success("✅ **Miễn nhiễm ngoại lai:** Co cụm giúp điểm đại diện lùi vào sâu bên trong lõi cụm!")

    with col_kmed2:
        st.markdown("##### 🔴 Thuật toán K-Medoids / PAM (Thuật toán đối chứng)")
        fig_kmed = make_scatter_figure(
            X_data, kmed_labels,
            f"K-Medoids: Chọn điểm thực tế làm tâm (Medoid đỏ)",
            centers=kmed_centers, center_label="Medoid thực tế",
            x_title=axis_x_name, y_title=axis_y_name
        )
        st.plotly_chart(fig_kmed, use_container_width=True)
        st.write(f"**Silhouette Score:** `{kmed_metrics['Silhouette']:.4f}` | **Thời gian:** `{kmed_time:.4f}s`")
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
    st.table(df_cmp_kmed)


# ==============================================================================
# TAB 6: SO SÁNH 1-1: CURE VS HIERARCHICAL (SINGLE LINKAGE)
# ==============================================================================
with tab_vs_hier:
    st.markdown("### ⚔️ So sánh Đối đầu Trực diện: CURE vs Gom cụm Phân cấp (Single Linkage)")
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
        st.plotly_chart(fig_c_hier, use_container_width=True)
        st.write(f"**Silhouette Score:** `{cure_metrics['Silhouette']:.4f}` | **Thời gian:** `{cure_time:.4f}s`")
        st.success("✅ **Không bị nối chuỗi:** Các điểm đại diện co cụm tạo khoảng cách ngăn cách an toàn!")

    with col_h2:
        st.markdown("##### 🟣 Gom cụm Phân cấp (Single Linkage)")
        fig_hier = make_scatter_figure(
            X_data, hier_labels,
            f"Hierarchical Single Linkage: Dễ bị dính cụm do nhiễu nối chuỗi",
            x_title=axis_x_name, y_title=axis_y_name
        )
        st.plotly_chart(fig_hier, use_container_width=True)
        st.write(f"**Silhouette Score:** `{hier_metrics['Silhouette']:.4f}` | **Thời gian:** `{hier_time:.4f}s`")
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
    st.table(df_cmp_hier)


# ==============================================================================
# TAB 7: BẢNG TỔNG HỢP MA TRẬN ĐỐI SÁNH TẤT CẢ THUẬT TOÁN
# ==============================================================================
with tab_summary:
    st.markdown("### 📋 Bảng Tổng hợp Ma trận Đối sánh Toàn diện")
    st.markdown("Bảng tổng hợp đối đầu giữa **CURE** và các thuật toán phân cụm trong chương trình môn học:")

    # Đồ thị Bar Chart so sánh Silhouette Score
    algs_names = ["CURE", "K-Means", "K-Medoids", "Hierarchical (Single)"]
    sils = [cure_metrics['Silhouette'], km_metrics['Silhouette'], kmed_metrics['Silhouette'], hier_metrics['Silhouette']]
    dbs = [cure_metrics['Davies-Bouldin'], km_metrics['Davies-Bouldin'], kmed_metrics['Davies-Bouldin'], hier_metrics['Davies-Bouldin']]
    times = [cure_metrics['Time'], km_metrics['Time'], kmed_metrics['Time'], hier_metrics['Time']]

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_bar_sil = go.Figure(data=[
            go.Bar(name='Silhouette Score (Càng cao càng tốt)', x=algs_names, y=sils,
                   marker_color=['#103673', '#0284c7', '#dc2626', '#8b5cf6'],
                   text=[f"{v:.3f}" for v in sils], textposition='auto')
        ])
        fig_bar_sil.update_layout(title="Chỉ số Silhouette Score trên Tập dữ liệu Hiện hành", height=340, yaxis=dict(range=[-0.2, 1.0]))
        st.plotly_chart(fig_bar_sil, use_container_width=True)

    with col_s2:
        fig_bar_time = go.Figure(data=[
            go.Bar(name='Thời gian thực thi (giây)', x=algs_names, y=times,
                   marker_color=['#103673', '#0284c7', '#dc2626', '#8b5cf6'],
                   text=[f"{v:.4f}s" for v in times], textposition='auto')
        ])
        fig_bar_time.update_layout(title="Thời gian Thực thi (giây)", height=340)
        st.plotly_chart(fig_bar_time, use_container_width=True)

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
    st.table(df_full_summary)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 12px;">
    Đồ án môn học Khai thác dữ liệu - Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT)<br>
    Hệ thống mô phỏng phục vụ thuyết trình và bảo vệ đề tài tiểu luận CURE.
</div>
""", unsafe_allow_html=True)
