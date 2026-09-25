"""Các thành phần giao diện dùng chung cho mọi tab của ứng dụng.

Chỉnh màu, khoảng cách, biểu đồ và định dạng chỉ số tại đây để các tab
không tự tạo những kiểu trình bày khác nhau.
"""

# %% Cell 1 - Thư viện và hằng số giao diện
from html import escape

import streamlit as st

CHART_HEIGHT = 480
PRIMARY_COLOR = "#103673"
CLUSTER_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#9333ea", "#0891b2"]


# %% Cell 2 - Kiểu trình bày chung (gọi một lần sau set_page_config)
def apply_theme():
    """Thống nhất tiêu đề, khung giải thích và ô chỉ số trên toàn bộ tab."""
    st.markdown("""
    <style>
      .stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }
      .stTabs h3, .stTabs h4, .stTabs h5 { color: #103673; }
      .app-tab-header { border-bottom: 1px solid #e2e8f0; margin-bottom: 20px; padding-bottom: 16px; }
      .app-tab-header h3 { margin: 0 0 8px; font-size: 1.35rem; }
      .app-tab-header p { color: #475569; margin: 0; line-height: 1.6; }
      .app-card { background: #f8fafc; border: 1px solid #e2e8f0;
        border-left: 4px solid #103673; border-radius: 8px;
        padding: 16px; margin: 12px 0; font-size: 14px; line-height: 1.7; color: #334155; }
      .app-card div, .app-card span, .app-card ul, .app-card ol,
      .app-card b { color: #334155 !important; font-size: inherit !important; }
      .stTabs [data-testid="stMetric"] { background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 12px; }
      .stTabs [data-testid="stMetricValue"] { font-size: 1.25rem; }
    </style>
    """, unsafe_allow_html=True)


# %% Cell 3 - Tiêu đề, chỉ số, biểu đồ và bảng
def render_tab_header(title, description, context):
    """Mọi tab bắt đầu bằng cùng một mẫu: tiêu đề → mục đích → dữ liệu."""
    st.markdown(
        f'<div class="app-tab-header"><h3>{escape(title)}</h3>'
        f'<p>{escape(description)}</p></div>', unsafe_allow_html=True,
    )
    st.caption(context)


def render_metrics(metrics):
    """Cùng thứ tự, độ chính xác và đơn vị cho các thuật toán đối sánh."""
    items = [
        ("Silhouette", f"{metrics['Silhouette']:.4f}"),
        ("Davies-Bouldin", f"{metrics['Davies-Bouldin']:.4f}"),
        ("Calinski-Harabasz", f"{metrics['Calinski-Harabasz']:.1f}"),
        ("Thời gian (giây)", f"{metrics['Time']:.4f}"),
    ]
    # Hai cột giúp chỉ số không bị chật trong một nửa màn hình so sánh.
    for start in (0, 2):
        for column, (label, value) in zip(st.columns(2), items[start:start + 2]):
            column.metric(label, value)


def render_chart(figure, key):
    """Áp dụng cùng kích thước và kiểu chữ ngay trước khi hiển thị."""
    figure.update_layout(
        height=CHART_HEIGHT,
        template="plotly_white",
        font=dict(family="Segoe UI, sans-serif", size=13, color="#334155"),
        title_font=dict(size=15, color=PRIMARY_COLOR),
        margin=dict(l=20, r=20, t=60, b=80),
        paper_bgcolor="white", plot_bgcolor="#fafbfc",
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=11)),
    )
    st.plotly_chart(figure, use_container_width=True, key=key)


def render_table(data):
    """Bảng thống nhất chiều rộng, ẩn cột chỉ mục và cho phép cuộn."""
    st.dataframe(data, use_container_width=True, hide_index=True)
