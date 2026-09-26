# Ghi nhận hoàn thiện mã nguồn CURE — 26/09/2026

## Phạm vi bàn giao

Sửa thuật toán, tiền xử lý, mô phỏng, đối sánh, diễn giải kết quả, tài liệu mã nguồn
và đóng gói bản chạy. Giữ menu 340px bên trái và 11 tác vụ. Các tài liệu Word/PowerPoint
có sẵn chưa được biên tập lại; cần cập nhật số liệu theo bảng mới trước khi nộp.

## Kiểm tra đã thực hiện

- 17 kiểm tra tự động Python đều đạt, gồm thuật toán, dữ liệu, UI và một bài gọi Node.js.
- Trong bài Canvas, 9 cấu hình JavaScript/Python khớp nhãn và chuỗi phép gom:
  ví dụ 6 điểm, bộ ngẫu nhiên, điểm trùng nhau; alpha = 0 / 0.4 / 1.
- CURE đối chiếu bản tham chiếu duyệt mọi cặp độc lập; kiểm tra lấy mẫu/gán điểm ngoài mẫu,
  refit, trung tâm/nhãn nhất quán, số đại diện và dữ liệu không hợp lệ.
- DIANA: hai điểm phải ra hai cụm; k=N; tọa độ trùng nhau; quy tắc chuyển từng điểm.
- PAM: chi phí giảm sau SWAP, không còn hoán đổi đơn nào cải thiện ở bộ thử,
  medoid là điểm đầu vào và xử lý điểm trùng nhau.
- Trình duyệt Edge headless: mở ứng dụng, menu 340px, chuyển tác vụ, tiến/lùi Canvas,
  kéo tới bước cuối, bản độc lập nhận CSV/thêm điểm/đổi tham số; màn hình 390px không tràn ngang.
- 540 lượt fit đo đạc hoàn tất: 4 bộ × 9 phương pháp × 5 seed × 3 lần lặp.
  Các lượt warm-up không tính trong 540 lượt; một luồng BLAS, perf_counter, không tính vẽ/metrics.

Đây là kiểm tra hồi quy và đối chiếu trên các bộ thử, không phải chứng minh hình thức
về mọi dữ liệu có thể có.

## Kết quả đáng lưu ý

ARI trung bình ± độ lệch chuẩn qua 5 seed (11,22,33,44,55), 200 điểm, chuẩn hóa,
k bằng số nhóm sinh dữ liệu; c=4, alpha=0.4; K-Means n_init=10;
DBSCAN eps=0.25, min_samples=5. Không dò tham số để chọn kết quả có lợi.

| Dữ liệu | CURE | K-Means | PAM | DBSCAN |
|---|---:|---:|---:|---:|
| Two Moons | 0.529 ± 0.215 | 0.474 ± 0.028 | 0.474 ± 0.032 | 0.780 ± 0.081 |
| Concentric Circles | 0.116 ± 0.026 | -0.005 ± 0.000 | -0.004 ± 0.001 | 0.572 ± 0.028 |
| Anisotropic Blobs | 0.733 ± 0.230 | 0.708 ± 0.293 | 0.716 ± 0.280 | 0.535 ± 0.052 |
| Blobs with Outliers | 0.400 ± 0.548 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.931 ± 0.064 |

Các kết quả này không ủng hộ khẳng định CURE luôn tốt hơn. CURE có thể nhạy với
c/alpha và hình dạng dữ liệu, đặc biệt vòng tròn đồng tâm hoặc điểm nhiễu làm tạo cụm nhỏ.
Bản cài đặt không có bước loại ngoại lai hai pha; khi nhiễu tạo cụm riêng, k cố định
có thể buộc các nhóm sạch bị gộp. Đây là giới hạn cần trình bày khi bảo vệ.

ARI/NMI ở bộ có ngoại lai tính trên nhãn thật sạch của cùng một tập điểm cho tất cả
thuật toán. Vì vậy ARI cao không chứng minh đã phát hiện hoặc loại bỏ ngoại lai.
Với DBSCAN, đọc thêm Noise fraction và Evaluated points khi xem các chỉ số nội tại.
Chi tiết 8 phương pháp chính và DBSCAN nằm trong benchmark_results/summary.csv.

## Cách thuyết trình đề xuất

1. Giải thích điểm đại diện và hệ số co bằng ví dụ 6 điểm (lịch sử thật, không ghi tay).
2. Thay đổi c/alpha, quan sát kết quả và thừa nhận cấu hình có thể thất bại.
3. Trình bày cùng dữ liệu, cùng chuẩn hóa và cùng k cho các phương pháp cần k.
4. Đọc ARI/NMI cùng chỉ số nội tại; tách nhận xét chất lượng khỏi tốc độ.
5. Nêu rõ mẫu khách hàng, mã hóa chi tiêu, PCA nếu có; không gọi nhóm tuổi cao là ngoại lai mặc định.
6. Kết luận phạm vi áp dụng và giới hạn, không tuyên bố ưu thế tổng quát.

## Tệp xuất

- runs.csv: tất cả lượt đo.
- summary.csv: thống kê qua trung bình từng seed; count là số seed có chỉ số hợp lệ.
- timings.csv: thời gian của toàn bộ lượt đo, gồm min/max/std/count.
- config.json: tham số, phiên bản thư viện, quy tắc tính chỉ số và phạm vi phép đo.
- toy_example_steps/: báo cáo và hình được tạo từ cùng thuật toán CURE.

Bản mã cũ trước sửa nằm trong .review/before_algorithm_completion.zip ở workspace.
Bản mới được đồng bộ vào BAO_CAO_NOP_GIANG_VIEN; gói CURE_HOAN_THIEN.zip chứa mã
và kết quả mới, không đóng gói môi trường ảo, cache hay tài liệu thuyết trình cũ.
