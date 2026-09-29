"""Ví dụ tính tay tương tác, phát lại lịch sử của thuật toán CURE thật."""
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import streamlit as st

from cure_algorithm import CURECluster
from data_pipeline import TOY_POINTS
from ui_components import CLUSTER_COLORS, render_table, render_tab_header


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


def make_toy_figure(points, groups, step):
    """Vẽ theo mẫu tính tay: khung trắng, nhãn tọa độ và đại diện dấu X đỏ."""
    fig = Figure(figsize=(8, 5), dpi=150, facecolor='white')
    ax = fig.subplots()
    for label, group in enumerate(groups.values()):
        pts = points[group['indices']]
        color = '#16a34a' if step == 0 else CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
        ax.scatter(pts[:, 0], pts[:, 1], s=65, color=color,
                   edgecolors='#166534' if step == 0 else color, zorder=3)
        mean = group['mean']
        ax.scatter(mean[0], mean[1], marker='*', s=170, color='#facc15',
                   edgecolors='black', linewidths=.8, zorder=4)
        reps = np.asarray(group['representatives'])
        ax.scatter(reps[:, 0], reps[:, 1], marker='x', s=85, color='red',
                   linewidths=1.8, zorder=5)
    for i, point in enumerate(points):
        ax.annotate(f'P{i+1}({point[0]:g},{point[1]:g})', point,
                    xytext=(7, 5), textcoords='offset points',
                    color='#103673', fontsize=10, fontweight='bold', zorder=6)
    ax.set(xlim=(0, 11), ylim=(0, 11), xticks=range(0, 11, 2), yticks=range(0, 11, 2))
    ax.set_xlabel('Tọa độ X', fontweight='bold')
    ax.set_ylabel('Tọa độ Y', fontweight='bold')
    ax.set_title(f'Bước {step}: ' + ('Khởi tạo 6 cụm đơn lẻ' if step == 0 else f'Sau sáp nhập — còn {len(groups)} cụm'),
                 color='#103673', fontweight='bold', pad=14)
    ax.grid(True, linestyle='--', linewidth=.7, color='#d1d5db')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#4b5563')
        spine.set_linewidth(1)
    fig.tight_layout()
    return fig


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
    with col_t1:
        fig = make_toy_figure(points, groups, step)
        st.pyplot(fig, use_container_width=True)
        st.caption('Điểm tròn: điểm dữ liệu · Sao vàng: trọng tâm · Dấu X đỏ: đại diện sau co.')
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
