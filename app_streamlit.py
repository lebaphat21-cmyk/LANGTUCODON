"""Hệ thống mô phỏng và đối sánh CURE. Chạy: streamlit run app_streamlit.py."""
# %% Thư viện
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
from cure_algorithm import CURE
from data_pipeline import DATASETS, FEATURES, TOY_POINTS, prepare_dataset
from evaluation import format_score
from experiments import ALGORITHMS, run_experiment
from visualization import render_canvas_html, replay_payload
from ui_components import (apply_theme, render_tab_header, render_metrics,
                           render_chart, render_table, CLUSTER_COLORS)

ROOT = Path(__file__).resolve().parent

# %% Hàm dùng chung
@st.cache_data(show_spinner='Đang chuẩn bị dữ liệu…')
def load_data(name, n, seed, pair, standardize, source):
    return prepare_dataset(name, n, seed, pair, standardize, source)


@st.cache_data(show_spinner='Đang tính kết quả phân cụm…')
def cached_experiment(name, X, k, c, alpha, seed, truth):
    return run_experiment(name, X, k, c, alpha, seed, truth)


@st.cache_data(show_spinner=False)
def elbow_scores(X, seed):
    rows = []
    for k in range(2, min(6, len(X)-1, len(np.unique(X, axis=0))) + 1):
        with threadpool_limits(limits=1):
            model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        labels = model.labels_
        sil = silhouette_score(X, labels) if 2 <= len(set(labels)) < len(X) else np.nan
        rows.append({'k': k, 'Inertia': model.inertia_, 'Silhouette': sil})
    return pd.DataFrame(rows, columns=['k', 'Inertia', 'Silhouette'])


def scatter_figure(data, model, title):
    labels = model.labels_
    frame = pd.DataFrame(data['plot'], columns=['X', 'Y'])
    frame['Cụm'] = [f'Cụm {i+1}' if i >= 0 else 'Nhiễu' for i in labels]
    colors = {f'Cụm {i+1}': CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in set(labels) if i >= 0}
    colors['Nhiễu'] = '#94a3b8'
    fig = px.scatter(frame, x='X', y='Y', color='Cụm', color_discrete_map=colors, title=title,
                     labels={'X': data['axes'][0], 'Y': data['axes'][1]})
    fig.update_traces(marker={'size': 7, 'opacity': .8})
    def original(points):
        return data['scaler'].inverse_transform(points) if data['scaler'] is not None else points
    if isinstance(model, CURE):
        for i, reps in enumerate(model.get_representatives()):
            pts = original(reps)
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode='markers',
                                    marker={'symbol': 'x', 'color': '#111827', 'size': 10},
                                    name='Đại diện đã co', showlegend=i == 0))
        means = original(model.get_cluster_means())
        fig.add_trace(go.Scatter(x=means[:, 0], y=means[:, 1], mode='markers',
                                marker={'symbol': 'star', 'color': '#eab308', 'size': 14}, name='Trọng tâm'))
    elif hasattr(model, 'cluster_centers_'):
        means = original(model.cluster_centers_)
        fig.add_trace(go.Scatter(x=means[:, 0], y=means[:, 1], mode='markers',
                                marker={'symbol': 'diamond', 'color': '#111827', 'size': 11},
                                name='Medoid' if hasattr(model, 'medoid_indices_') else 'Trọng tâm'))
    return fig


def cluster_profile(data, model):
    if data['frame'] is None:
        frame = pd.DataFrame(data['plot'], columns=data['axes'])
    else:
        frame = data['frame'][['Age', 'Spending_Score_Num', 'Work_Experience', 'Family_Size']].copy()
    frame['Cụm'] = model.labels_ + 1
    grouped = frame.groupby('Cụm')
    table = grouped.mean().round(3)
    table.insert(0, 'Số điểm', grouped.size())
    table.insert(1, 'Tỷ lệ (%)', (100 * grouped.size() / len(frame)).round(1))
    return table.reset_index()


def show_result(name, key):
    model, metrics = result(name)
    render_chart(scatter_figure(data, model, name), key)
    render_metrics(metrics)
    st.caption(f'Số cụm thực tế: {metrics["Clusters"]} · Thời gian một lượt fit, dùng lại từ cache khi đầu vào giữ nguyên.')
    if data['truth'] is not None:
        st.caption(f'ARI: {format_score(metrics["ARI"])} · NMI: {format_score(metrics["NMI"])} (so với nhãn thật).')
    return model, metrics


def compare(name):
    left, right = st.columns(2)
    with left:
        _, first = show_result('CURE', f'compare_cure_{name}')
    with right:
        _, second = show_result(name, f'compare_other_{name}')
    delta = first['Silhouette'] - second['Silhouette']
    if np.isfinite(delta):
        st.info(f'Trên dữ liệu hiện tại, chênh lệch Silhouette CURE − {name}: {delta:+.4f}. '
                'Đây là một tiêu chí đánh giá, không phải kết luận thuật toán nào luôn tốt hơn.')
    else:
        st.info('Silhouette không xác định cho ít nhất một kết quả; không xếp hạng theo chỉ số này.')
    st.caption('Nhãn và màu “Cụm 1” giữa hai thuật toán không bảo đảm cùng một nhóm điểm.')

# %% Trang và điều khiển
st.set_page_config(page_title='CURE Clustering Visualizer - HUIT', page_icon='🔬', layout='wide')
apply_theme()
st.markdown('''<div style="background:linear-gradient(135deg,#103673,#1f4e79);padding:22px;border-radius:12px;margin-bottom:20px;color:white">
<div style="color:#fbbf24;font-weight:700">TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP. HỒ CHÍ MINH (HUIT)</div>
<div style="font-size:22px;font-weight:700;margin-top:8px">HỆ THỐNG MÔ PHỎNG & ĐỐI SÁNH THUẬT TOÁN PHÂN CỤM CURE</div>
<div style="margin-top:6px;font-size:13px">Khám phá cách gom cụm · Quan sát điểm đại diện · Kiểm chứng bằng thực nghiệm</div></div>''', unsafe_allow_html=True)
st.sidebar.header('⚙️ Cấu hình thực nghiệm')
dataset_type = st.sidebar.selectbox('Tập dữ liệu', DATASETS, key='dataset')
customer = dataset_type == DATASETS[0]
pair = st.sidebar.selectbox('Thuộc tính phân cụm', list(FEATURES), key='features') if customer else list(FEATURES)[0]
uploaded = st.sidebar.file_uploader('CSV khách hàng khác (tùy chọn)', type=['csv']) if customer else None
n_samples = st.sidebar.slider('Số điểm trong thực nghiệm', 20, 600, 140 if customer else 300, 20, key='sample_count')
seed = int(st.sidebar.number_input('Random seed', 0, 9999, 42, key='seed'))
standardize = st.sidebar.checkbox('Chuẩn hóa các thuộc tính', value=True, key='standardize',
                                   help='Mỗi thuộc tính có trung bình 0, độ lệch chuẩn 1. PCA luôn chuẩn hóa trước khi chiếu.')
k = st.sidebar.slider('Số cụm mục tiêu (k)', 2, 6, 3 if customer else 2, key='k')
c = st.sidebar.slider('Số đại diện mỗi cụm (c)', 1, 12, 4, key='c')
alpha = st.sidebar.slider('Hệ số co (alpha)', 0., 1., .4, .05, key='alpha',
                          help='0: giữ vị trí điểm đại diện; 1: co về trọng tâm. Giá trị phù hợp tùy dữ liệu.')
if customer:
    try:
        source = uploaded.getvalue() if uploaded else (ROOT / 'Data' / 'Test.csv').read_bytes()
    except FileNotFoundError:
        st.error('Không tìm thấy Data/Test.csv. Hãy tải CSV khách hàng ở thanh bên trái.')
        st.stop()
else:
    source = None
try:
    data = load_data(dataset_type, n_samples, seed, pair, standardize, source)
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()
X = data['X']
if len(X) < k:
    st.error(f'Chỉ có {len(X)} điểm; hãy giảm k xuống không lớn hơn số điểm.')
    st.stop()
if len(np.unique(X, axis=0)) < k:
    st.warning('Số tọa độ phân biệt nhỏ hơn k; một số cụm trùng vị trí và chỉ số chất lượng có thể không có ý nghĩa.')
for note in data['notes']:
    st.sidebar.caption(note)
st.sidebar.caption(f'{data["preprocessing"]}. Biểu đồ kết quả dùng đơn vị hiển thị gốc; khoảng cách dùng dữ liệu mô hình.')
with st.sidebar.expander('Gợi ý k từ K-Means'):
    if st.checkbox('Tính gợi ý', key='show_elbow'):
        elbow = elbow_scores(X, seed)
        if elbow.empty or elbow.Silhouette.notna().sum() == 0:
            st.info('Không đủ dữ liệu để gợi ý k.')
        else:
            best = int(elbow.loc[elbow.Silhouette.idxmax(), 'k'])
            st.line_chart(elbow.set_index('k')['Inertia'])
            st.dataframe(elbow, hide_index=True)
            st.caption(f'k={best} có Silhouette K-Means cao nhất trong các giá trị thử; không phải k tối ưu đã xác nhận cho CURE.')


def result(name):
    # Tham số không liên quan không làm mất cache của các thuật toán đối chứng.
    return cached_experiment(name, X, k, c if name == 'CURE' else 4,
                             alpha if name == 'CURE' else .4, seed, data['truth'])


context = f'{dataset_type} | N={len(X)} / {data["total"]} | k={k}, c={c}, α={alpha} | seed={seed} | {data["preprocessing"]}'
TASK_OPTIONS = {
    'tab_cure_sim': '🎮 Mô phỏng tương tác CURE',
    'tab_cure_main': '🎯 CURE: Kết quả phân cụm',
    'tab_cure_steps': '📝 CURE: Ví dụ tính tay',
    'tab_cure_flow': '🔍 CURE: Quy trình 5 giai đoạn',
    'tab_vs_kmeans': '⚔️ CURE vs K-Means',
    'tab_vs_kmedoids': '⚔️ CURE vs K-Medoids (PAM)',
    'tab_vs_hier': '⚔️ CURE vs Hierarchical',
    'tab_vs_agnes': '🔗 AGNES: So sánh 4 linkage',
    'tab_vs_diana': '✂️ CURE vs DIANA',
    'tab_summary': '📋 Tổng hợp đối sánh',
    'tab_outlier': '🔎 Khảo sát nhóm tuổi & ngoại lai',
}
with st.container(key='task_navigation'):
    selected_task = st.selectbox('☰ Chọn tác vụ', list(TASK_OPTIONS), format_func=TASK_OPTIONS.get, key='selected_task')

# %% Mô phỏng
if selected_task == 'tab_cure_sim':
    render_tab_header('🎮 Mô phỏng từng bước CURE', 'Chạy, tạm dừng, lùi bước hoặc kéo thanh tiến trình để quan sát phép gom.', context)
    model, _ = result('CURE')
    st.iframe(render_canvas_html(X, model), height=790)
    st.caption('Mô phỏng dùng đúng lịch sử của kết quả Python. Để thử điểm tự tạo, dùng bản Canvas độc lập trong web_demo.')
    st.download_button('Tải lịch sử gom cụm (JSON)', json.dumps(replay_payload(X, model), ensure_ascii=False, indent=2),
                       'cure_history.json', 'application/json')

# %% Kết quả CURE
if selected_task == 'tab_cure_main':
    render_tab_header('🎯 Kết quả phân cụm CURE', 'Điểm đại diện, trọng tâm và thành viên lấy từ cùng một kết quả gom cụm.', context)
    model, metrics = show_result('CURE', 'cure_main')
    render_table(cluster_profile(data, model))
    st.caption('Tên cụm là nhãn kỹ thuật; diễn giải khách hàng cần dựa trên thống kê và hiểu biết nghiệp vụ.')
    export = data['frame'].copy() if customer else pd.DataFrame(data['plot'], columns=data['axes'])
    export['CURE_cluster'] = model.labels_ + 1
    st.download_button('Tải dữ liệu và nhãn CURE (CSV)', export.to_csv(index=False).encode('utf-8-sig'), 'cure_labels.csv', 'text/csv')

# %% Ví dụ tính tay
if selected_task == 'tab_cure_steps':
    render_tab_header('📝 Ví dụ 6 điểm', 'Mỗi dòng được tạo từ lịch sử thực thi; trường hợp hòa chọn cặp có ID nhỏ hơn.', 'Dữ liệu cố định | k=2, c=2, α=0.5 | không chuẩn hóa')
    toy = CURE(2, 2, .5).fit(TOY_POINTS)
    render_table(pd.DataFrame(TOY_POINTS, columns=['X', 'Y']).assign(Điểm=[f'P{i+1}' for i in range(6)]))
    st.latex(r'r^{\prime}=r+\alpha(\mu-r),\qquad d(C_i,C_j)=\min_{r_i,r_j}\|r_i-r_j\|_2')
    st.iframe(render_canvas_html(TOY_POINTS, toy, 'Ví dụ tính tay: 6 điểm'), height=790)
    for i, event in enumerate(toy.history_, 1):
        with st.expander(f'Bước {i}: gom cụm {event["left"]+1} và {event["right"]+1} · d={event["distance"]:.4f}'):
            st.write('Trọng tâm:', event['mean'])
            render_table(pd.DataFrame(event['representatives'], columns=['Rep X sau co', 'Rep Y sau co']))

# %% Phạm vi và lý thuyết
if selected_task == 'tab_cure_flow':
    render_tab_header('🔍 Quy trình CURE và phạm vi cài đặt', 'Phân biệt thuật toán dữ liệu lớn trong tài liệu gốc với thực nghiệm trên mẫu.', context)
    render_table(pd.DataFrame([
        ['1. Lấy mẫu', 'Có', 'App chọn mẫu chung cho mọi thuật toán; lớp CURE còn hỗ trợ sample_size.'],
        ['2. Phân hoạch và gom sơ bộ', 'Chưa triển khai', 'Không dùng để tuyên bố hiệu năng dữ liệu lớn.'],
        ['3. Gom bằng điểm đại diện', 'Có', 'Chọn xa nhất, co về mean, chọn cặp gần nhất bằng heap.'],
        ['4. Loại ngoại lai hai pha', 'Chưa triển khai', 'Kháng ảnh hưởng của ngoại lai không đồng nghĩa phát hiện hoặc loại bỏ ngoại lai.'],
        ['5. Gán điểm ngoài mẫu', 'Có trong lớp CURE', 'Giữ nhãn mẫu; gán các điểm còn lại theo đại diện gần nhất. App đối sánh trên mẫu chung.'],
    ], columns=['Giai đoạn', 'Trạng thái', 'Giải thích']))
    st.latex(r'r^{\prime}=r+\alpha(\mu-r)')
    st.write('Với số đại diện c và số chiều cố định, bản dùng heap có thời gian O(s² log s), bộ nhớ O(s²). '
             'Gán điểm ngoài mẫu tốn O((N−s)kcd). Bản này chưa được thiết kế cho dữ liệu hàng triệu điểm.')
    st.markdown('[Bài báo CURE — Guha, Rastogi, Shim, SIGMOD 1998](https://www2.cs.sfu.ca/CourseCentral/459/han/papers/guha98.pdf)')

# %% Các đối sánh
if selected_task == 'tab_vs_kmeans':
    render_tab_header('⚔️ CURE và K-Means', 'K-Means dùng 10 lần khởi tạo; cả hai dùng cùng dữ liệu mô hình và k.', context)
    compare('K-Means')

if selected_task == 'tab_vs_kmedoids':
    render_tab_header('⚔️ CURE và K-Medoids (PAM)', 'PAM thực hiện BUILD và SWAP; medoid là điểm dữ liệu thực tế.', context)
    compare('K-Medoids (PAM)')
    model, _ = result('K-Medoids (PAM)')
    st.caption(f'PAM: {model.n_iter_} lượt kiểm tra hoán đổi · hội tụ: {model.converged_} · tổng khoảng cách: {model.inertia_:.4f}.')

if selected_task == 'tab_vs_hier':
    render_tab_header('⚔️ CURE và Hierarchical Single Linkage', 'Single Linkage cũng chính là AGNES với linkage=single.', context)
    compare('AGNES (single)')

if selected_task == 'tab_vs_agnes':
    render_tab_header('🔗 Bốn biến thể AGNES', 'Single, complete, average và Ward khác nhau ở tiêu chí gom cụm.', context)
    rows = []
    for name in ['AGNES (single)', 'AGNES (complete)', 'AGNES (average)', 'AGNES (ward)']:
        model, metrics = result(name)
        render_chart(scatter_figure(data, model, name), name)
        rows.append({'Thuật toán': name, **metrics})
    render_table(pd.DataFrame(rows))
    st.caption('Ward giảm tổng bình phương sai lệch trong cụm; Single có thể nối cụm qua các điểm cầu nối. Kết quả tùy dữ liệu.')

if selected_task == 'tab_vs_diana':
    render_tab_header('✂️ CURE và DIANA', 'DIANA bắt đầu từ một cụm và tách dần theo đường kính, chuyển từng điểm có chênh lệch lớn nhất.', context)
    compare('DIANA')

# %% Tổng hợp
if selected_task == 'tab_summary':
    render_tab_header('📋 Tổng hợp đối sánh', 'Tám phương pháp trên cùng đầu vào; không đếm Single Linkage hai lần.', context)
    rows = [{'Thuật toán': name, **result(name)[1]} for name in ALGORITHMS]
    summary = pd.DataFrame(rows)
    render_table(summary)
    render_chart(px.bar(summary, x='Thuật toán', y='Silhouette', title='Silhouette trên dữ liệu hiện tại'), 'summary_sil')
    if data['truth'] is not None:
        render_chart(px.bar(summary, x='Thuật toán', y='ARI', title='ARI so với nhãn thật (cao hơn là tốt hơn)'), 'summary_ari')
    st.info('Silhouette và Calinski–Harabasz thường ưu ái cụm lồi. Dùng ARI/NMI khi có nhãn thật; '
            'không suy ra CURE luôn thắng từ một chỉ số hoặc một bộ dữ liệu.')
    st.caption('N/A/ô trống: chỉ số không xác định. Với dữ liệu có nhiễu nhân tạo, ARI/NMI tính trên cùng các điểm có nhãn thật khác −1. '
               'Thời gian trên app là một lượt fit; hãy dùng thực nghiệm lặp để so hiệu năng.')
    settings = {'dataset': dataset_type, 'n': len(X), 'seed': seed, 'k': k, 'c': c, 'alpha': alpha,
                'preprocessing': data['preprocessing'], 'feature_pair': pair if customer else None}
    st.download_button('Tải bảng đối sánh (CSV)', summary.to_csv(index=False).encode('utf-8-sig'), 'comparison.csv', 'text/csv')
    st.download_button('Tải cấu hình (JSON)', json.dumps(settings, ensure_ascii=False, indent=2), 'experiment.json', 'application/json')
    with st.expander('Thực nghiệm lặp nhiều seed'):
        st.write('Script benchmark chạy 4 bộ tổng hợp, 8 phương pháp và DBSCAN; xuất kết quả từng lượt, trung bình, độ lệch chuẩn và tỷ lệ nhiễu.')
        st.code('python benchmark_comparison.py --seeds 11 22 33 44 55 --repeats 3 --samples 200')
        st.caption('Đây là thực nghiệm theo cấu hình công khai; DBSCAN chưa được dò tham số tối ưu. Không coi bảng này là xếp hạng phổ quát.')

# %% Khảo sát tuổi
if selected_task == 'tab_outlier':
    render_tab_header('🔎 Khảo sát nhóm tuổi và ngoại lai', 'Thống kê theo ngưỡng nghiệp vụ; không tự xem tuổi cao là ngoại lai.', context)
    if not customer:
        st.info('Chọn dữ liệu khách hàng để khảo sát nhóm tuổi. Với dữ liệu tổng hợp, xem ARI/NMI trong bảng đối sánh.')
    else:
        threshold = st.slider('Ngưỡng tuổi khảo sát', 40, 100, 70, key='age_threshold')
        frame = data['frame'].copy()
        older = frame.Age >= threshold
        count = int(older.sum())
        st.write(f'Trong mẫu **{len(frame)}** người đang phân tích, có **{count}** người từ **{threshold} tuổi** '
                 f'(**{100*count/len(frame):.1f}%**). Tổng dữ liệu nguồn: **{data["total"]}** dòng.')
        frame['Nhóm tuổi'] = np.where(older, f'≥ {threshold}', f'< {threshold}')
        fig = px.histogram(frame, x='Age', color='Nhóm tuổi', nbins=25, title='Phân bố tuổi trong mẫu')
        fig.add_vline(x=threshold, line_dash='dash')
        render_chart(fig, 'age_distribution')
        table = frame.groupby('Nhóm tuổi').agg(Số_người=('Age', 'size'), Tuổi_TB=('Age', 'mean'),
                Chi_tiêu_TB=('Spending_Score_Num', 'mean'), Kinh_nghiệm_TB=('Work_Experience', 'mean'))
        render_table(table.reset_index())
        if count == 0 or count == len(frame):
            st.info('Một nhóm không có người trong mẫu; chưa thể đối chiếu hai nhóm.')
        else:
            difference = frame.loc[older, 'Spending_Score_Num'].mean() - frame.loc[~older, 'Spending_Score_Num'].mean()
            st.write(f'Chênh lệch điểm chi tiêu trung bình (nhóm cao tuổi − nhóm còn lại): **{difference:+.3f}** '
                     'trên thang mã hóa 1–3. Đây là mô tả mẫu, không phải quan hệ nhân quả.')
        q1, q3 = frame.Age.quantile([.25, .75])
        lower, upper = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
        flags = (frame.Age < lower) | (frame.Age > upper)
        st.caption(f'Tham khảo quy tắc IQR riêng cho tuổi: ngưỡng [{lower:.1f}, {upper:.1f}], '
                   f'{flags.sum()} điểm bị gắn cờ. Cờ IQR không phải nhãn ngoại lai do CURE tạo ra.')
        model, _ = result('CURE')
        frame['Cụm CURE'] = model.labels_ + 1
        render_table(pd.crosstab(frame['Cụm CURE'], frame['Nhóm tuổi']).reset_index())

# %% Chân trang
st.divider()
st.caption('CURE demo học thuật · Cấu hình, dữ liệu và tiêu chí đánh giá quyết định kết luận. '
           'Bản cài đặt chưa có phân hoạch/lọc ngoại lai hai pha.')
