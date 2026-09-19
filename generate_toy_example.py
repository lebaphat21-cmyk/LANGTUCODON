"""
Xây dựng và minh họa chi tiết Ví dụ tính tay từng bước (Toy Example) của thuật toán CURE
Dành riêng cho Báo cáo và Slide thuyết trình ĐH Công Thương TP.HCM (HUIT)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

os.makedirs('toy_example_steps', exist_ok=True)

# 6 điểm 2D minh họa
points = np.array([
    [1.0, 2.0],  # P1
    [2.0, 3.0],  # P2
    [2.0, 1.0],  # P3
    [8.0, 7.0],  # P4
    [9.0, 8.0],  # P5
    [8.0, 9.0]   # P6
])
point_labels = ['P1(1,2)', 'P2(2,3)', 'P3(2,1)', 'P4(8,7)', 'P5(9,8)', 'P6(8,9)']
c_param = 2      # Số điểm đại diện
alpha = 0.5      # Hệ số co cụm
k_target = 2     # Số cụm đích

def draw_step(step_num, title, clusters, active_ids, desc):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx_color, cid in enumerate(active_ids):
        cl = clusters[cid]
        pts = cl['points']
        color = colors[idx_color % len(colors)]
        
        # Vẽ các điểm dữ liệu
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=120, edgecolors='black', zorder=4, label=f"Cụm {cl['name']}")
        for pt in pts:
            # Tìm nhãn gốc
            p_idx = np.where(np.all(np.isclose(points, pt), axis=1))[0][0]
            ax.annotate(point_labels[p_idx], xy=(pt[0], pt[1]), xytext=(pt[0]+0.15, pt[1]+0.15),
                        fontsize=10, fontweight='bold', color='#103673')
            
        # Vẽ trọng tâm
        mean = cl['mean']
        ax.scatter(mean[0], mean[1], color='yellow', marker='*', s=200, edgecolors='black', zorder=6)
        
        # Vẽ điểm đại diện đã co
        reps = cl['reps']
        ax.scatter(reps[:, 0], reps[:, 1], color='red', marker='x', s=100, linewidths=2.5, zorder=7)
        
        # Vẽ đường nối giữa rep và mean nếu có nhiều điểm
        if len(pts) > 1:
            for rep in reps:
                ax.plot([mean[0], rep[0]], [mean[1], rep[1]], 'r--', alpha=0.5)

    ax.set_title(f"BƯỚC {step_num}: {title.upper()}", fontsize=12, fontweight='bold', color='#103673', pad=12)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.set_xlabel('Tọa độ X', fontweight='bold')
    ax.set_ylabel('Tọa độ Y', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Text box mô tả
    plt.figtext(0.15, 0.02, desc, wrap=True, horizontalalignment='left', fontsize=9.5,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F9F9F9', edgecolor='#CCCCCC'))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    out_path = os.path.join('toy_example_steps', f"step_{step_num}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

report_lines = []
report_lines.append("=== CHI TIẾT BÀI TOÁN TÍNH TAY TỪNG BƯỚC (TOY EXAMPLE) ===")
report_lines.append("Tập dữ liệu 6 điểm 2D:")
for i, l in enumerate(point_labels):
    report_lines.append(f"  - {l}")
report_lines.append(f"Tham số: Số điểm đại diện c = {c_param}, Hệ số co cụm alpha = {alpha}, Số cụm đích k = {k_target}\n")

# BƯỚC 0: Khởi tạo
clusters = {}
for i in range(len(points)):
    pt = points[i:i+1]
    clusters[i] = {
        'name': f"P{i+1}",
        'points': pt,
        'mean': pt[0],
        'reps': pt
    }
active_ids = list(range(6))
draw_step(0, "Khởi tạo 6 cụm ban đầu (Mỗi điểm là 1 cụm)", clusters, active_ids,
          "Mỗi điểm ban đầu là 1 cụm. Trọng tâm mean và điểm đại diện rep trùng với chính điểm đó.")

# BƯỚC 1: Sáp nhập P1 và P2 (d = sqrt(2) ~ 1.414)
# Cụm mới C12
p12 = np.vstack([points[0], points[1]])
mean12 = np.mean(p12, axis=0) # (1.5, 2.5)
# 2 rep points co cụm:
rep12_raw = p12
rep12_shrunk = rep12_raw + alpha * (mean12 - rep12_raw) # (1.25, 2.25) và (1.75, 2.75)
clusters[0] = {
    'name': '{P1, P2}',
    'points': p12,
    'mean': mean12,
    'reps': rep12_shrunk
}
active_ids.remove(1)
del clusters[1]

report_lines.append("--- BƯỚC 1: Sáp nhập P1 và P2 thành cụm C{1,2} ---")
report_lines.append(f"Khoảng cách nhỏ nhất ban đầu: d(P1, P2) = sqrt((2-1)^2 + (3-2)^2) = sqrt(2) ≈ 1.414")
report_lines.append(f"Trọng tâm cụm mới: mean = ((1+2)/2, (2+3)/2) = ({mean12[0]}, {mean12[1]})")
report_lines.append(f"Vì cụm có 2 điểm (<= c=2), chọn cả 2 điểm làm điểm đại diện thô.")
report_lines.append(f"Áp dụng công thức co cụm với alpha = 0.5: p' = p + alpha * (mean - p)")
report_lines.append(f"  + p1' = (1, 2) + 0.5 * ((1.5, 2.5) - (1, 2)) = ({rep12_shrunk[0][0]}, {rep12_shrunk[0][1]})")
report_lines.append(f"  + p2' = (2, 3) + 0.5 * ((1.5, 2.5) - (2, 3)) = ({rep12_shrunk[1][0]}, {rep12_shrunk[1][1]})\n")

draw_step(1, "Sáp nhập P1 và P2 thành cụm {P1, P2}", clusters, active_ids,
          f"Khoảng cách d(P1, P2)=1.414 nhỏ nhất -> Sáp nhập.\nTrọng tâm mean=(1.5, 2.5). Hai điểm đại diện co về mean thành (1.25, 2.25) và (1.75, 2.75).")

# BƯỚC 2: Sáp nhập P4 và P5 (d = 1.414)
p45 = np.vstack([points[3], points[4]])
mean45 = np.mean(p45, axis=0) # (8.5, 7.5)
rep45_shrunk = p45 + alpha * (mean45 - p45) # (8.25, 7.25) và (8.75, 7.75)
clusters[3] = {
    'name': '{P4, P5}',
    'points': p45,
    'mean': mean45,
    'reps': rep45_shrunk
}
active_ids.remove(4)
del clusters[4]

report_lines.append("--- BƯỚC 2: Sáp nhập P4 và P5 thành cụm C{4,5} ---")
report_lines.append(f"Khoảng cách nhỏ nhất tiếp theo: d(P4, P5) = sqrt((9-8)^2 + (8-7)^2) = sqrt(2) ≈ 1.414")
report_lines.append(f"Trọng tâm cụm mới: mean = ({mean45[0]}, {mean45[1]})")
report_lines.append(f"Hai điểm đại diện sau khi co alpha = 0.5: ({rep45_shrunk[0][0]}, {rep45_shrunk[0][1]}) và ({rep45_shrunk[1][0]}, {rep45_shrunk[1][1]})\n")

draw_step(2, "Sáp nhập P4 và P5 thành cụm {P4, P5}", clusters, active_ids,
          f"Khoảng cách d(P4, P5)=1.414 nhỏ nhất -> Sáp nhập.\nTrọng tâm mean=(8.5, 7.5). Điểm đại diện co về mean thành (8.25, 7.25) và (8.75, 7.75).")

# BƯỚC 3: Sáp nhập C{1,2} và P3
# d(P3, rep1') = sqrt((2-1.25)^2 + (1-2.25)^2) = sqrt(0.5625 + 1.5625) = sqrt(2.125) ~ 1.458
p123 = np.vstack([p12, points[2]])
mean123 = np.mean(p123, axis=0) # (5/3, 2.0) ~ (1.667, 2.0)
# Cụm có 3 điểm, c=2. Farthest point heuristic:
# Điểm xa mean nhất:
dists_m = np.linalg.norm(p123 - mean123, axis=1)
first_rep = p123[np.argmax(dists_m)] # P2(2, 3) hoặc P3(2, 1)
# Điểm thứ hai: xa điểm thứ nhất nhất
dists_r1 = np.linalg.norm(p123 - first_rep, axis=1)
second_rep = p123[np.argmax(dists_r1)]
rep123_raw = np.array([first_rep, second_rep])
rep123_shrunk = rep123_raw + alpha * (mean123 - rep123_raw)

clusters[0] = {
    'name': '{P1, P2, P3}',
    'points': p123,
    'mean': mean123,
    'reps': rep123_shrunk
}
active_ids.remove(2)
del clusters[2]

report_lines.append("--- BƯỚC 3: Sáp nhập cụm C{1,2} và điểm P3 thành cụm C{1,2,3} ---")
report_lines.append(f"Tính khoảng cách từ P3(2, 1) tới các rep của C{1,2}:")
report_lines.append(f"  + d(P3, p1') = sqrt((2-1.25)^2 + (1-2.25)^2) = sqrt(2.125) ≈ 1.458")
report_lines.append(f"  + d(P3, p2') = sqrt((2-1.75)^2 + (1-2.75)^2) = sqrt(3.125) ≈ 1.768")
report_lines.append(f"  => dist(C{1,2}, P3) = min(1.458, 1.768) = 1.458 (nhỏ nhất hiện tại)")
report_lines.append(f"Trọng tâm cụm mới C{1,2,3}: mean = (1.667, 2.000)")
report_lines.append(f"Vì cụm có 3 điểm > c=2, thuật toán chọn 2 điểm phân tán tốt nhất:")
report_lines.append(f"  + Điểm 1 (xa mean nhất): {first_rep}")
report_lines.append(f"  + Điểm 2 (xa điểm 1 nhất): {second_rep}")
report_lines.append(f"Co 2 điểm này về mean với alpha = 0.5: ({rep123_shrunk[0][0]:.3f}, {rep123_shrunk[0][1]:.3f}) và ({rep123_shrunk[1][0]:.3f}, {rep123_shrunk[1][1]:.3f})\n")

draw_step(3, "Sáp nhập C{1,2} và P3 thành cụm {P1, P2, P3}", clusters, active_ids,
          f"Khoảng cách dist(C{{1,2}}, P3) = 1.458 -> Sáp nhập.\nCụm có 3 điểm (> c=2), chọn 2 điểm rải rác xa nhau nhất rồi co về mean.")

# BƯỚC 4: Sáp nhập C{4,5} và P6 thành C{4,5,6}
p456 = np.vstack([p45, points[5]])
mean456 = np.mean(p456, axis=0) # (25/3, 8.0) ~ (8.333, 8.0)
dists_m456 = np.linalg.norm(p456 - mean456, axis=1)
first_rep456 = p456[np.argmax(dists_m456)]
dists_r456 = np.linalg.norm(p456 - first_rep456, axis=1)
second_rep456 = p456[np.argmax(dists_r456)]
rep456_raw = np.array([first_rep456, second_rep456])
rep456_shrunk = rep456_raw + alpha * (mean456 - rep456_raw)

clusters[3] = {
    'name': '{P4, P5, P6}',
    'points': p456,
    'mean': mean456,
    'reps': rep456_shrunk
}
active_ids.remove(5)
del clusters[5]

report_lines.append("--- BƯỚC 4: Sáp nhập cụm C{4,5} và điểm P6 thành cụm C{4,5,6} ---")
report_lines.append(f"Khoảng cách dist(C{4,5}, P6) = 1.458 -> Sáp nhập thành C{4,5,6}.")
report_lines.append(f"Trọng tâm cụm mới: mean = (8.333, 8.000)")
report_lines.append(f"Điểm đại diện sau co cụm alpha = 0.5: ({rep456_shrunk[0][0]:.3f}, {rep456_shrunk[0][1]:.3f}) và ({rep456_shrunk[1][0]:.3f}, {rep456_shrunk[1][1]:.3f})\n")
report_lines.append("=== KẾT THÚC: SỐ CỤM CÒN LẠI LÀ 2 (ĐẠT K_TARGET = 2) ===")
report_lines.append("Kết quả phân cụm:")
report_lines.append("  - Cụm 1: {P1, P2, P3} (Cụm bên trái)")
report_lines.append("  - Cụm 2: {P4, P5, P6} (Cụm bên phải)")
report_lines.append(f"Khoảng cách giữa 2 cụm cuối cùng rất lớn (dist ≈ 6.5 > 1.458) -> Thuật toán dừng lại chuẩn xác!")

draw_step(4, "Hoàn tất phân cụm: Đạt số cụm đích k = 2", clusters, active_ids,
          "Khoảng cách dist(C{4,5}, P6) = 1.458 -> Sáp nhập.\nSố cụm còn lại đúng bằng k=2 -> THUẬT TOÁN DỪNG VÀ XUẤT KẾT QUẢ!")

with open(os.path.join('toy_example_steps', 'toy_example_report.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("Toy example complete! Images saved to toy_example_steps/")
