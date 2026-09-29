"""Ví dụ tính tay tương tác, phát lại lịch sử của thuật toán CURE thật."""
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cure_algorithm import CURE
from data_pipeline import TOY_POINTS
from ui_components import CLUSTER_COLORS, render_chart, render_table, render_tab_header


def build_steps(points):
    model = CURE(2, 2, .5).fit(points)
    groups = {i: dict(indices=[i], mean=p.tolist(), representatives=[p.tolist()])
              for i, p in enumerate(points)}
    states = [groups.copy()]
    for event in model.history_:
        left, right = event['left'], event['right']
        groups = groups.copy()
        members = groups[left]['indices'] + groups[right]['indices']
        del groups[right]
        groups[left] = dict(indices=members, mean=event['mean'],
                            representatives=event['representatives'])
        states.append(groups)
    return model.history_, states


def group_name(group):
    return '{' + ', '.join(f'P{i+1}' for i in group['indices']) + '}'


def render_toy_example():
    render_tab_header('📝 CURE: Ví dụ tính tay với 6 điểm tùy chọn',
                      'Nhập tọa độ của từng điểm để tính lại biểu đồ và các bước gom cụm.',
                      'N = 6 | k = 2 | c = 2 | α = 0.5 | Tọa độ gốc, không chuẩn hóa')
    st.caption('Có thể nhập tọa độ âm, thập phân hoặc các điểm trùng nhau. Khi hòa khoảng cách, chọn cặp ID nhỏ nhất.')
    if st.button('Khôi phục 6 điểm mẫu', key='toy_reset'):
        for i, point in enumerate(TOY_POINTS):
            for axis, value in zip(('x', 'y'), point):
                st.session_state[f'toy_{i}_{axis}'] = float(value)
    points = []
    columns = st.columns(3)
    for i, default in enumerate(TOY_POINTS):
        with columns[i % 3]:
            st.markdown(f'**P{i+1}**')
            points.append([st.number_input(f'{axis.upper()} của P{i+1}', value=float(value),
                                           key=f'toy_{i}_{axis}', format='%.4f')
                           for axis, value in zip(('x', 'y'), default)])
    points = np.asarray(points, dtype=float)
    if not np.isfinite(points).all():
        st.error('Vui lòng nhập tọa độ là số hữu hạn cho cả 6 điểm.')
        return
    history, states = build_steps(points)
    titles = ['Bước 0: Khởi tạo 6 cụm']
    for i, event in enumerate(history):
        before = states[i]
        titles.append(f"Bước {i+1}: Gom {group_name(before[event['left']])} và "
                      f"{group_name(before[event['right']])}")
    step = st.radio('Chọn bước thực hiện để quan sát:', range(len(states)),
                    format_func=lambda i: titles[i], horizontal=True, key='toy_step')
    groups = states[step]
    fig = go.Figure()
    for label, group in enumerate(groups.values()):
        ids = group['indices']
        pts = points[ids]
        color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode='markers+text',
                                text=[f'P{i+1}' for i in ids], textposition='top center',
                                name=group_name(group), marker=dict(size=12, color=color)))
        reps = np.asarray(group['representatives'])
        fig.add_trace(go.Scatter(x=reps[:, 0], y=reps[:, 1], mode='markers',
                                name='Đại diện sau co', showlegend=label == 0,
                                marker=dict(symbol='x', size=12, color='black')))
        fig.add_trace(go.Scatter(x=[group['mean'][0]], y=[group['mean'][1]], mode='markers',
                                name='Trọng tâm', showlegend=label == 0,
                                marker=dict(symbol='star', size=15, color='#eab308')))
    fig.update_layout(title=titles[step], xaxis_title='X', yaxis_title='Y')
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    render_chart(fig, key='toy_chart')
    st.markdown('#### 📐 Công thức và phép tính chi tiết')
    st.latex(r'd(A,B)=\min_{p\in R_A,q\in R_B}\|p-q\|_2,\quad m=\frac{1}{|C|}\sum_{p\in C}p,\quad r=p+0.5(m-p)')
    if step:
        event = history[step-1]
        before = states[step-1]
        a, b = before[event['left']], before[event['right']]
        rows = []
        for p in a['representatives']:
            for q in b['representatives']:
                distance = np.linalg.norm(np.asarray(p)-q)
                rows.append({'Đại diện A': str(p), 'Đại diện B': str(q),
                             'Phép tính Euclidean': f'√[({p[0]:.4f} − {q[0]:.4f})² + ({p[1]:.4f} − {q[1]:.4f})²]',
                             'Khoảng cách': distance})
        render_table(pd.DataFrame(rows))
        st.markdown(f"**Khoảng cách nhỏ nhất của cặp được gom: {event['distance']:.6f}.**")
        merged = groups[event['left']]
        members = points[merged['indices']]
        mean = np.asarray(merged['mean'])
        for axis in range(2):
            expression = ' + '.join(f'({p[axis]:.4f})' for p in members)
            st.markdown(f"m{'ₓ' if axis == 0 else 'ᵧ'} = ({expression}) / {len(members)} = **{mean[axis]:.6f}**")
        st.markdown('Chọn điểm xa trọng tâm nhất; điểm tiếp theo có khoảng cách tới đại diện đã chọn lớn nhất. Nếu cụm có tối đa 2 điểm, chọn tất cả.')
        for rep in np.asarray(merged['representatives']):
            original = 2 * rep - mean
            st.markdown(f'r = ({original[0]:.4f}, {original[1]:.4f}) + 0.5 × '
                        f'[({mean[0]:.4f}, {mean[1]:.4f}) − ({original[0]:.4f}, {original[1]:.4f})] '
                        f'= **({rep[0]:.6f}, {rep[1]:.6f})**')
    st.markdown('#### Khoảng cách giữa các cụm sau bước hiện tại')
    rows = []
    for a, b in combinations(groups.values(), 2):
        ra, rb = np.asarray(a['representatives']), np.asarray(b['representatives'])
        rows.append({'Cụm A': group_name(a), 'Cụm B': group_name(b),
                     'Khoảng cách CURE': np.linalg.norm(ra[:, None, :] - rb[None, :, :], axis=2).min()})
    render_table(pd.DataFrame(rows))
    if step == len(history):
        st.success('Hoàn tất: còn đúng k = 2 cụm. Kết quả được tính từ 6 điểm bạn đã nhập.')
    else:
        st.info(f'Bước tiếp theo: {titles[step+1]}; khoảng cách nhỏ nhất = {history[step]["distance"]:.6f}.')
