"""Ví dụ tính tay tương tác, phát lại lịch sử của thuật toán CURE thật."""
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cure_algorithm import CURECluster
from data_pipeline import TOY_POINTS
from ui_components import CLUSTER_COLORS, render_chart, render_table, render_tab_header


def group_distance(a, b):
    ra, rb = np.asarray(a['representatives']), np.asarray(b['representatives'])
    return float(np.linalg.norm(ra[:, None, :] - rb[None, :, :], axis=2).min())


def build_steps(points, choices=()):
    groups = {i: dict(indices=[i], mean=p.tolist(), representatives=[p.tolist()])
              for i, p in enumerate(points)}
    states, history = [groups.copy()], []
    while len(groups) > 2:
        pairs = list(combinations(sorted(groups), 2))
        nearest = min(pairs, key=lambda pair: (group_distance(groups[pair[0]], groups[pair[1]]), pair))
        pair = choices[len(history)] if len(history) < len(choices) else nearest
        if pair not in pairs:
            raise ValueError('Chọn hai cụm khác nhau đang tồn tại ở bước này.')
        left, right = pair
        members = groups[left]['indices'] + groups[right]['indices']
        cluster = CURECluster(points[members], left, members)
        cluster.update_representatives(2, .5)
        event = dict(left=left, right=right, distance=group_distance(groups[left], groups[right]),
                     nearest=nearest, minimum=group_distance(groups[nearest[0]], groups[nearest[1]]),
                     mean=cluster.mean.tolist(), representatives=cluster.rep_points.tolist())
        history.append(event)
        groups = groups.copy()
        del groups[right]
        groups[left] = dict(indices=members, mean=event['mean'], representatives=event['representatives'])
        states.append(groups)
    return history, states


def select_merge(step):
    # Capture the numeric step when rendering the selector. Another widget's
    # session value may be stale or unset while Streamlit runs callbacks.
    choices = st.session_state['toy_choices'][:step-1]
    # Preserve the preceding computed steps before overriding this merge.
    history, _ = build_steps(TOY_POINTS, choices)
    choices = [(e['left'], e['right']) for e in history[:step-1]]
    st.session_state['toy_choices'] = choices + [st.session_state[f'toy_pair_{step}']]



def group_name(group):
    return '{' + ', '.join(f'P{i+1}' for i in group['indices']) + '}'


def render_toy_example():
    render_tab_header('📝 Bài toán Ví dụ Tính tay Từng bước (Toy Example)',
                      'Theo dõi phép tính trên 6 điểm mẫu và tự chọn điểm/cụm muốn gom ở mỗi bước.',
                      'Ví dụ: N = 6 | k = 2 | c = 2 | α = 0.5')
    st.markdown(r"""
    Tập dữ liệu minh họa gồm **6 điểm 2D cố định** như bản trước:
    * Cụm bên trái: $P_1(1, 2)$, $P_2(2, 3)$, $P_3(2, 1)$.
    * Cụm bên phải: $P_4(8, 7)$, $P_5(9, 8)$, $P_6(8, 9)$.
    * **Cấu hình:** $k = 2$, $c = 2$, hệ số co $\alpha = 0.5$.
    """)
    st.caption('Chọn bước 1–4, rồi chọn hai điểm/cụm cần gom. Các bước phía sau được tính lại; mặc định chọn cặp gần nhất theo CURE.')
    if st.button('Khôi phục cách gom mẫu', key='toy_reset'):
        st.session_state['toy_choices'] = []
        st.session_state['toy_step_v2'] = 0
        for i in range(1, 5):
            st.session_state.pop(f'toy_pair_{i}', None)
    st.session_state.setdefault('toy_choices', [])
    points = TOY_POINTS
    history, states = build_steps(points, st.session_state['toy_choices'])
    titles = ['Bước 0: Khởi tạo 6 cụm']
    for i, event in enumerate(history):
        before = states[i]
        titles.append(f"Bước {i+1}: Gom {group_name(before[event['left']])} và "
                      f"{group_name(before[event['right']])}")
    step = st.radio('Chọn bước thực hiện để quan sát:', list(range(len(states))),
                    format_func=lambda i: 'Bước 0: Khởi tạo 6 cụm' if i == 0 else f'Bước {i}: Gom còn {6-i} cụm',
                    horizontal=True, key='toy_step_v2')
    if step:
        before = states[step-1]
        event = history[step-1]
        pairs = list(combinations(sorted(before), 2))
        selected_pair = (event['left'], event['right'])
        st.session_state[f'toy_pair_{step}'] = selected_pair
        st.selectbox('Chọn hai điểm/cụm để gom ở bước này:', pairs,
                     format_func=lambda pair: f'{group_name(before[pair[0]])} + {group_name(before[pair[1]])}',
                     key=f'toy_pair_{step}', on_change=select_merge, args=(step,))
        if not np.isclose(event['distance'], event['minimum'], rtol=1e-12, atol=1e-12):
            st.warning(f"Bạn đang thử gom cặp có khoảng cách {event['distance']:.6f}; "
                       f"CURE sẽ chọn cặp gần nhất với khoảng cách {event['minimum']:.6f}. "
                       'Đây là lựa chọn tính tay của bạn, khác quy tắc chọn cặp của CURE.')
    col_t1, col_t2 = st.columns(2)
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
    with col_t1:
        render_chart(fig, key='toy_chart')
    with col_t2:
        st.markdown('#### 📐 Công thức và phép tính chi tiết')
        st.markdown('##### 📘 1. Công thức gốc lý thuyết')
        st.latex(r'd(A,B)=\min_{p\in R_A,q\in R_B}\|p-q\|_2,\quad m=\frac{1}{|C|}\sum_{p\in C}p,\quad r=p+0.5(m-p)')
        st.markdown('##### 🔢 2. Phép tính số chi tiết')
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
            st.markdown(f"**Khoảng cách giữa hai cụm được chọn: {event['distance']:.6f}.**")
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
        else:
            st.markdown('Khởi tạo 6 cụm đơn lẻ; mỗi điểm là đại diện duy nhất của cụm đó.')
            st.latex(r'd(P_1,P_2)=\sqrt{(2-1)^2+(3-2)^2}=\sqrt{2}\approx1.414214')
            st.markdown('Hai cặp P1–P2 và P4–P5 cùng có khoảng cách nhỏ nhất. Mặc định chọn P1–P2 theo ID; bạn có thể chọn cặp khác ở Bước 1.')
    st.markdown('#### Khoảng cách giữa các cụm sau bước hiện tại')
    rows = []
    for a, b in combinations(groups.values(), 2):
        ra, rb = np.asarray(a['representatives']), np.asarray(b['representatives'])
        rows.append({'Cụm A': group_name(a), 'Cụm B': group_name(b),
                     'Khoảng cách CURE': np.linalg.norm(ra[:, None, :] - rb[None, :, :], axis=2).min()})
    render_table(pd.DataFrame(rows))
    if step == len(history):
        st.success('Hoàn tất: còn đúng k = 2 cụm. Kết quả theo các cặp điểm/cụm đã chọn.')
    else:
        st.info(f'Bước tiếp theo: {titles[step+1]}; khoảng cách cặp được gom = {history[step]["distance"]:.6f}.')
