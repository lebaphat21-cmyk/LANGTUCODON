# Hệ thống mô phỏng & đối sánh thuật toán phân cụm CURE

Bản hoàn thiện mã nguồn: 26/09/2026. Ứng dụng giữ 11 tác vụ trong menu nhỏ bên trái.
Đây là mô hình học thuật trên dữ liệu số nhỏ/vừa; không phải bản CURE dữ liệu lớn đầy đủ.

## Khởi chạy

Python 3.12 trở lên. Chạy tại thư mục chứa README này:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app_streamlit.py
```

`requirements.lock.txt` ghi toàn bộ phiên bản đã kiểm thử. Có thể thay `requirements.txt`
bằng file khóa này để tái lập đúng môi trường. Trên máy đã cài môi trường, mở `run_demo.bat`.
Bản độc lập `web_demo/index.html` mở trực tiếp bằng trình duyệt, không cần server/CDN.

## Cách thực nghiệm

1. Chọn dữ liệu, số điểm và seed. CSV khách hàng cần các cột `Age`, `Spending_Score`,
   `Work_Experience`, `Family_Size`. Có thể tải CSV khác ngay cả khi đã có Test.csv.
2. Bật/tắt chuẩn hóa. PCA luôn chuẩn hóa 4 thuộc tính trước khi chiếu 2D.
3. Đặt k, c, alpha. Các thuật toán đối chứng dùng cùng dữ liệu mô hình và k.
4. Xem mô phỏng rồi kết quả CURE. Các phép gom trong Canvas lấy trực tiếp từ lịch sử Python.
5. Xem đối sánh; tải nhãn, bảng chỉ số và cấu hình để lưu lại thực nghiệm.
6. Chạy benchmark lặp để báo cáo hiệu năng thay vì dùng thời gian một lượt trên app.

```powershell
python -m unittest discover -s tests -v
python benchmark_comparison.py --seeds 11 22 33 44 55 --repeats 3 --samples 200
python generate_toy_example.py
python export_individual_charts.py
```

Trong môi trường ảo, thay `python` bằng `.venv\Scripts\python.exe` nếu chưa activate.

## Những điểm đã sửa

- CURE giữ chỉ số thành viên cụm mẫu; chỉ gán điểm ngoài mẫu bằng đại diện gần nhất.
  Trọng tâm, đại diện, nhãn và lịch sử gom nhất quán. Refit xóa lịch sử cũ.
- Heap chọn cặp gần nhất thay cho quét toàn bộ ma trận con mỗi bước.
  Cặp hết hiệu lực bị loại bằng phiên bản cụm; trường hợp hòa chọn ID nhỏ hơn.
- Chọn đại diện không chọn lại cùng chỉ số khi tọa độ trùng nhau.
- PAM có BUILD và SWAP; không gọi cập nhật medoid luân phiên là PAM.
- DIANA chuyển từng điểm có chênh lệch lớn nhất, giữ nhóm chính không rỗng;
  đường kính luôn tính chính xác, kể cả dữ liệu trên 200 điểm.
- Dữ liệu mô hình tách khỏi tọa độ hiển thị; không thêm nhiễu vào dữ liệu phân cụm.
  Biểu đồ Plotly trả đại diện về đơn vị gốc; Canvas phát lịch sử trong không gian mô hình.
- Không có fallback dữ liệu âm thầm khi CSV lỗi. Kiểm tra cột, số dòng và giá trị hữu hạn;
  điền trung vị/mode trong mẫu và thông báo số giá trị đã điền.
- Nhận xét khách hàng dựa trên thống kê hiện tại, không suy diễn vai trò cụm từ số nhãn.
  Tuổi cao là nhóm khảo sát; cờ IQR riêng cho tuổi không được gọi là nhãn CURE.
- N/A (NaN trong CSV) thay cho các giá trị giả -1/99/0 khi chỉ số không xác định.
- Thêm ARI/NMI với nhãn thật; benchmark báo tỷ lệ nhiễu, số điểm đánh giá,
  nhiều seed/lần lặp, warm-up, perf_counter và cấu hình/phiên bản đầy đủ.
- Giao diện chỉ tính các thuật toán cần hiển thị, có cache riêng cho từng phương pháp.
  Chỉ tính gợi ý k K-Means khi người dùng bật; không gọi đó là k tối ưu cho CURE.

## Tổ chức mã

| File | Trách nhiệm |
|---|---|
| `cure_algorithm.py` | CURE, PAM, DIANA và kiểm tra dữ liệu đầu vào |
| `data_pipeline.py` | Nạp CSV, dữ liệu tổng hợp có nhãn thật, chuẩn hóa, PCA |
| `evaluation.py` | Chỉ số chất lượng, mặt nạ đánh giá, định dạng N/A |
| `experiments.py` | Một nguồn cấu hình và đo fit cho app/benchmark |
| `app_streamlit.py` | Điều khiển và 11 tác vụ; không chứa bản JavaScript tính CURE riêng |
| `ui_components.py` | Menu 340px, bảng, biểu đồ, chỉ số và kiểu trình bày |
| `visualization.py`, `templates/cure_replay.html` | Phát lại lịch sử CURE, tiến/lùi và thanh bước |
| `web_demo/` | Sandbox độc lập, thêm điểm hoặc CSV; không lấy thời gian để so Python |
| `benchmark_comparison.py` | Thực nghiệm lặp và kết quả trong benchmark_results |
| `generate_toy_example.py` | Tạo ví dụ tính tay từ lịch sử thuật toán thật |
| `export_individual_charts.py` | Xuất 4 hình đối sánh cấu hình hiện hành |
| `tests/` | Kiểm thử thuật toán, dữ liệu và tương tác ứng dụng |

## Phạm vi và cách diễn giải

CURE đã có lấy mẫu, gom bằng đại diện co và gán phần ngoài mẫu. Chưa có phân hoạch,
loại ngoại lai hai pha, k-d tree hoặc xử lý dữ liệu ngoài bộ nhớ của bài báo gốc.
App lấy một mẫu chung để so công bằng; kết quả trên app không đại diện cho toàn bộ CSV.

Với c và số chiều d cố định, CURE dùng heap có thời gian O(s² log s), bộ nhớ O(s²).
Gán điểm ngoài mẫu O((N−s)kcd), xử lý theo lô để tránh ma trận khoảng cách quá lớn.
Sandbox JavaScript duyệt cặp có O(n³) thời gian; phép chiếu chỉ dùng để vẽ.
Nút đổi tham số trong sandbox khởi tạo lại để tránh trộn hai cấu hình trong một lần chạy.

Silhouette/Calinski–Harabasz ưu ái cụm lồi; không dùng một chỉ số để tuyên bố CURE
luôn tốt hơn. ARI/NMI của dữ liệu tổng hợp có nhiễu dùng cùng tập điểm có nhãn thật
khác -1 cho mọi phương pháp; nhãn dự đoán -1 vẫn được giữ trên tập điểm này.
Chỉ số nội tại của DBSCAN loại điểm dự đoán là nhiễu, do đó phải đọc cùng tỷ lệ nhiễu
và số điểm đánh giá. ARI cao trên các điểm sạch không đồng nghĩa phát hiện nhiễu tốt.

Benchmark cố định c=4, alpha=0.4, DBSCAN eps=0.25/min_samples=5, K-Means n_init=10;
đây không phải kết quả tối ưu hóa tham số của từng thuật toán. PAM BUILD xác định,
seed ảnh hưởng dữ liệu nhưng không ảnh hưởng khởi tạo PAM. Độ lệch chuẩn chất lượng
được tính qua trung bình mỗi seed, không coi các lần lặp thời gian là mẫu độc lập.
Chi tiêu Low/Average/High được mã hóa thứ bậc 1/2/3, không phải số tiền thực tế.

## Kết quả và tài liệu

- `benchmark_results/runs.csv`: 540 lượt đo, gồm 9 phương pháp × 4 bộ × 5 seed × 3 lặp.
- `benchmark_results/summary.csv`: trung bình, độ lệch chuẩn và số seed hợp lệ.
- `benchmark_results/timings.csv`: phân bố thời gian của toàn bộ lượt đo.
- `benchmark_results/config.json`: cấu hình, cách đo, phiên bản và giới hạn.
- `HOAN_THIEN.md`: ghi nhận kiểm thử và các kết quả cần lưu ý khi thuyết trình.
- `toy_example_steps/`: hình và báo cáo 6 điểm đã tạo lại.

Các Word/PowerPoint có sẵn chưa được biên soạn lại. Số liệu và khẳng định trong chúng
cần đối chiếu kết quả mới trước khi nộp. Một số hình cũ trong `charts/` được giữ để
không làm hỏng tài liệu cũ; chỉ 4 hình `moons/circles/aniso/outliers_comparison.png`
đã được tạo lại. Dùng `benchmark_results/` làm nguồn số liệu hiện hành.

## Tài liệu tham chiếu

- Guha, Rastogi, Shim (1998), CURE: https://www2.cs.sfu.ca/CourseCentral/459/han/papers/guha98.pdf
- PAM: https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/pam.html
- DIANA: https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/diana.html
- Đánh giá phân cụm: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
