"""Tạo ví dụ tính tay từ cùng lịch sử CURE mà ứng dụng sử dụng."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cure_algorithm import CURE
from data_pipeline import TOY_POINTS


def generate_report(output='toy_example_steps'):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    model = CURE(2, 2, .5).fit(TOY_POINTS)
    groups = {i: dict(indices=[i], mean=point, representatives=[point]) for i, point in enumerate(TOY_POINTS)}
    lines = ['CURE: 6 điểm | k=2 | c=2 | alpha=0.5 | thang đo gốc',
             'Trường hợp hòa: chọn cặp ID nhỏ nhất.', 'Điểm: ' + repr(TOY_POINTS.tolist())]
    for step in range(len(model.history_)+1):
        if step:
            e = model.history_[step-1]
            members = groups[e['left']]['indices'] + groups[e['right']]['indices']
            del groups[e['right']]
            groups[e['left']] = dict(indices=members, mean=e['mean'], representatives=e['representatives'])
            lines += [f'Bước {step}: gom {e["left"]+1} + {e["right"]+1}; d={e["distance"]:.6f}',
                      f'Thành viên: {[i+1 for i in members]}; mean={e["mean"]}',
                      f'Đại diện sau co: {e["representatives"]}']
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, group in enumerate(groups.values()):
            pts = TOY_POINTS[group['indices']]
            ax.scatter(*pts.T, label=f'Cụm {label+1}', s=65)
            reps = np.array(group['representatives'])
            ax.scatter(*reps.T, marker='x', c='red')
            ax.scatter(*np.array(group['mean']), marker='*', c='gold', edgecolor='black', s=140)
        for i, p in enumerate(TOY_POINTS):
            ax.annotate(f'P{i+1}', p, xytext=(6,6), textcoords='offset points')
        ax.set_title(f'CURE — bước {step}'); ax.set_aspect('equal'); ax.legend()
        fig.tight_layout(); fig.savefig(out/f'step_{step}.png', dpi=180); plt.close(fig)
    (out/'toy_example_report.txt').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    generate_report()
