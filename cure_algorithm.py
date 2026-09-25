"""
Thuật toán phân cụm CURE (Clustering Using REpresentatives)
Cài đặt tối ưu hóa cho môn Khai phá dữ liệu - ĐH Công Thương TP.HCM (HUIT)
"""

# %% Cell 01 - Thư viện tính toán
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

# %% Cell 02 - K-Medoids: khởi tạo → gán cụm → cập nhật medoid
class KMedoids:
    """
    Thuật toán K-Medoids (PAM - Partitioning Around Medoids) chuẩn môn Khai phá dữ liệu.
    Chọn medoid là điểm thực tế trong tập dữ liệu có tổng khoảng cách tới các điểm khác là nhỏ nhất.
    """
    def __init__(self, n_clusters=2, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoid_indices_ = None
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = len(X)
        rng = np.random.RandomState(self.random_state)
        medoids = rng.choice(n_samples, size=self.n_clusters, replace=False)
        dist_mat = cdist(X, X)

        for _ in range(self.max_iter):
            labels = np.argmin(dist_mat[:, medoids], axis=1)
            new_medoids = np.copy(medoids)

            for k in range(self.n_clusters):
                cluster_members = np.where(labels == k)[0]
                if len(cluster_members) > 0:
                    sub_dist = dist_mat[np.ix_(cluster_members, cluster_members)]
                    costs = np.sum(sub_dist, axis=1)
                    best_member = cluster_members[np.argmin(costs)]
                    new_medoids[k] = best_member

            if np.array_equal(medoids, new_medoids):
                break
            medoids = new_medoids

        self.medoid_indices_ = medoids
        self.labels_ = np.argmin(dist_mat[:, medoids], axis=1)
        self.cluster_centers_ = X[medoids]
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


# %% Cell 03 - Cấu trúc cụm và chọn điểm đại diện
class CURECluster:
    """Đại diện cho một cụm trong thuật toán CURE"""
    def __init__(self, points, cluster_id):
        self.cluster_id = cluster_id
        self.points = np.array(points, dtype=float)
        if len(self.points.shape) == 1:
            self.points = self.points.reshape(1, -1)
        self.mean = np.mean(self.points, axis=0)
        self.rep_points = np.copy(self.points)
        
    def update_representatives(self, c, alpha):
        """
        1. Tính trọng tâm mean của cụm.
        2. Chọn c điểm đại diện rải rác tốt nhất (well-scattered) bằng Farthest-Point Heuristic.
        3. Co các điểm đại diện về phía trọng tâm theo hệ số alpha:
           p' = p + alpha * (mean - p)
        """
        self.mean = np.mean(self.points, axis=0)
        n_points = len(self.points)
        
        if n_points <= c:
            selected_rep = np.copy(self.points)
        else:
            # Điểm đầu tiên: xa trọng tâm nhất
            dists_to_mean = np.linalg.norm(self.points - self.mean, axis=1)
            first_idx = np.argmax(dists_to_mean)
            selected_rep = [self.points[first_idx]]
            
            # Các điểm tiếp theo: chọn điểm có min khoảng cách tới các rep đã chọn là lớn nhất
            for _ in range(1, c):
                current_reps = np.array(selected_rep)
                dists = cdist(self.points, current_reps)
                min_dists = np.min(dists, axis=1)
                next_idx = np.argmax(min_dists)
                selected_rep.append(self.points[next_idx])
                
            selected_rep = np.array(selected_rep)
            
        # Co về phía trọng tâm
        self.rep_points = selected_rep + alpha * (self.mean - selected_rep)


# %% Cell 04 - CURE: lấy mẫu → gom cụm → gán nhãn
class CURE:
    """
    Lớp triển khai thuật toán CURE tối ưu tốc độ với ma trận khoảng cách động.
    
    Tham số:
    - n_clusters (k): Số cụm mục tiêu (mặc định 2)
    - n_representatives (c): Số điểm đại diện trên mỗi cụm (mặc định 5)
    - shrink_factor (alpha): Hệ số co cụm về trọng tâm (mặc định 0.5)
    - sample_size (s): Kích thước mẫu ngẫu nhiên (nếu None thì lấy toàn bộ)
    """
    def __init__(self, n_clusters=2, n_representatives=5, shrink_factor=0.5, sample_size=None, random_state=42):
        self.n_clusters = n_clusters
        self.n_representatives = n_representatives
        self.shrink_factor = shrink_factor
        self.sample_size = sample_size
        self.random_state = random_state
        self.clusters_ = []
        self.labels_ = None
        self.history_ = []

    def _cluster_dist(self, c1, c2):
        """Tính khoảng cách nhỏ nhất giữa các điểm đại diện (đã co) của 2 cụm"""
        dists = cdist(c1.rep_points, c2.rep_points)
        return np.min(dists)

    def fit(self, X):
        """Thực thi thuật toán CURE trên tập dữ liệu X"""
        # Bản demo dùng ma trận khoảng cách O(s²), chưa cài phân hoạch
        # hay hai pha loại ngoại lai trong quy trình CURE mở rộng.
        X = np.asarray(X, dtype=float)
        n_samples = len(X)
        
        # 1. Lấy mẫu ngẫu nhiên nếu kích thước dữ liệu lớn
        if self.sample_size is not None and self.sample_size < n_samples:
            rng = np.random.RandomState(self.random_state)  # Dùng instance, tránh side effect global seed
            sample_indices = rng.choice(n_samples, size=self.sample_size, replace=False)
            X_sample = X[sample_indices]
        else:
            X_sample = X
            
        N = len(X_sample)
        
        # Khởi tạo mỗi điểm là 1 cụm ban đầu
        clusters = {}
        for i in range(N):
            c = CURECluster(X_sample[i:i+1], cluster_id=i)
            c.update_representatives(self.n_representatives, self.shrink_factor)
            clusters[i] = c
            
        # Ma trận khoảng cách ban đầu giữa các điểm N x N
        # Vì ban đầu mỗi cụm là 1 điểm, dist_matrix là khoảng cách euclidean giữa các điểm
        d_condensed = pdist(X_sample)
        dist_matrix = squareform(d_condensed)
        np.fill_diagonal(dist_matrix, np.inf)
        
        active_ids = list(range(N))
        
        # 2. Gom cụm phân cấp tối ưu (duy trì ma trận khoảng cách)
        while len(active_ids) > self.n_clusters:
            # Tìm cặp cụm (u, v) có khoảng cách nhỏ nhất trong active_ids
            # Sub-matrix của active_ids
            sub_dist = dist_matrix[np.ix_(active_ids, active_ids)]
            
            # Tọa độ min trong sub_dist
            min_pos = np.argmin(sub_dist)
            r, c_idx = np.unravel_index(min_pos, sub_dist.shape)
            
            u = active_ids[r]
            v = active_ids[c_idx]
            
            if u > v:
                u, v = v, u  # Đảm bảo u < v
                
            # Sáp nhập cụm v vào cụm u
            c_u = clusters[u]
            c_v = clusters[v]
            merged_pts = np.vstack((c_u.points, c_v.points))
            new_cluster = CURECluster(merged_pts, cluster_id=u)
            new_cluster.update_representatives(self.n_representatives, self.shrink_factor)
            clusters[u] = new_cluster
            
            # Xóa cụm v
            del clusters[v]
            active_ids.remove(v)
            
            # Đánh dấu khoảng cách đến v là vô cực
            dist_matrix[v, :] = np.inf
            dist_matrix[:, v] = np.inf
            
            # Cập nhật lại khoảng cách từ cụm mới u đến các cụm còn lại trong active_ids
            for w in active_ids:
                if w == u:
                    dist_matrix[u, w] = np.inf
                    dist_matrix[w, u] = np.inf
                else:
                    d = self._cluster_dist(clusters[u], clusters[w])
                    dist_matrix[u, w] = d
                    dist_matrix[w, u] = d
                    
        self.clusters_ = [clusters[idx] for idx in active_ids]
        
        # 3. Gán nhãn toàn bộ dữ liệu X dựa vào điểm đại diện gần nhất
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        """Gán mỗi điểm vào cụm có điểm đại diện gần nhất"""
        X = np.asarray(X, dtype=float)
        
        all_reps = []
        rep_cluster_mapping = []
        for cluster_idx, c in enumerate(self.clusters_):
            for rep in c.rep_points:
                all_reps.append(rep)
                rep_cluster_mapping.append(cluster_idx)
                
        all_reps = np.array(all_reps)
        rep_cluster_mapping = np.array(rep_cluster_mapping)
        
        dists = cdist(X, all_reps)
        closest_rep_indices = np.argmin(dists, axis=1)
        return rep_cluster_mapping[closest_rep_indices]

    def get_representatives(self):
        """Danh sách các điểm đại diện của các cụm"""
        return [np.copy(c.rep_points) for c in self.clusters_]

    def get_cluster_means(self):
        """Danh sách trọng tâm các cụm"""
        return np.array([c.mean for c in self.clusters_])


# %% Cell 05 - DIANA: chọn cụm → tách cụm → gán nhãn
class DIANA:
    """
    DIANA (DIvisive ANAlysis) — Phân cụm phân cấp hướng từ trên xuống (Top-down Divisive).

    Nguyên lý hoạt động (ngược với AGNES/CURE):
      1. Bắt đầu với 1 cụm lớn duy nhất chứa tất cả N điểm.
      2. Chọn cụm có ĐƯỜNG KÍNH (max pairwise distance) lớn nhất để tách.
      3. Trong cụm đó, tìm điểm có avg-dissimilarity cao nhất tới các điểm còn lại
         → đó là "hạt nhân" (splinter) của cụm con mới.
      4. Lần lượt chuyển các điểm gần splinter hơn main-cluster sang cụm con.
      5. Lặp lại bước 2–4 đến khi đạt đúng k cụm mục tiêu.

    Tham số:
      - n_clusters (k): Số cụm mục tiêu (mặc định 2).
      - max_points_full: Ngưỡng kích thước cụm để tính diameter đầy đủ (default 200).
        Nếu cụm lớn hơn, dùng xấp xỉ (random sampling) để tránh quá chậm.
    """

    def __init__(self, n_clusters=2, max_points_full=200):
        self.n_clusters = n_clusters
        self.max_points_full = max_points_full
        self.labels_ = None
        self.clusters_ = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _diameter(self, X, indices):
        """Đường kính cụm = max khoảng cách giữa 2 điểm bất kỳ trong cụm."""
        pts = X[indices]
        if len(pts) <= 1:
            return 0.0
        # Nếu cụm nhỏ đủ: tính chính xác
        if len(pts) <= self.max_points_full:
            return float(np.max(pdist(pts)))
        # Cụm lớn: xấp xỉ bằng max khoảng cách tới trọng tâm * 2
        mean_pt = pts.mean(axis=0)
        return float(np.max(np.linalg.norm(pts - mean_pt, axis=1))) * 2.0

    def _split_cluster(self, X, indices):
        """
        Tách 1 cụm thành 2 theo cơ chế DIANA:
          a. Tìm điểm có avg dissimilarity cao nhất → "splinter" (hạt nhân cụm con).
          b. Di chuyển các điểm từ main → split nếu gần split hơn main.
          c. Lặp đến khi không còn điểm nào di chuyển.

        Trả về: (indices_main, indices_split)
        """
        pts = X[indices]
        n = len(pts)

        if n <= 1:
            return list(indices), []

        # a. Tính avg distance mỗi điểm đến các điểm khác trong cụm
        dist_mat = squareform(pdist(pts))  # n×n, đường chéo = 0
        # avg dissimilarity (bỏ qua chính nó)
        np.fill_diagonal(dist_mat, 0.0)
        avg_diss = dist_mat.sum(axis=1) / max(n - 1, 1)

        splinter_local = int(np.argmax(avg_diss))

        # Khởi tạo: main = tất cả trừ splinter, split = {splinter}
        main_set  = set(range(n)) - {splinter_local}
        split_set = {splinter_local}

        # b. Lặp di chuyển
        changed = True
        while changed:
            changed = False
            to_move = []
            for i in list(main_set):
                # avg dist đến main (trừ chính i)
                main_others = list(main_set - {i})
                if main_others:
                    d_main = dist_mat[i, main_others].mean()
                else:
                    d_main = np.inf  # main chỉ còn mình i → nên chuyển

                # avg dist đến split
                d_split = dist_mat[i, list(split_set)].mean()

                if d_split < d_main:
                    to_move.append(i)

            if to_move:
                for i in to_move:
                    main_set.discard(i)
                    split_set.add(i)
                changed = True

        indices_main  = [indices[i] for i in sorted(main_set)]
        indices_split = [indices[i] for i in sorted(split_set)]
        return indices_main, indices_split

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X):
        """Chạy DIANA trên tập dữ liệu X (ndarray shape N×D)."""
        X = np.asarray(X, dtype=float)
        n = len(X)

        # Bắt đầu: 1 cụm chứa tất cả
        clusters = [list(range(n))]

        while len(clusters) < self.n_clusters:
            # Chọn cụm có đường kính lớn nhất để tách
            diameters = [self._diameter(X, c) for c in clusters]
            split_idx = int(np.argmax(diameters))

            target = clusters[split_idx]
            if len(target) <= 1:
                # Không thể tách thêm → dừng
                break

            main_part, split_part = self._split_cluster(X, target)

            if not main_part or not split_part:
                # Tránh tách rỗng
                break

            clusters.pop(split_idx)
            clusters.append(main_part)
            clusters.append(split_part)

        # Gán nhãn
        self.labels_ = np.zeros(n, dtype=int)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = label

        self.clusters_ = clusters
        return self

    def fit_predict(self, X):
        """Fit và trả về nhãn phân cụm."""
        return self.fit(X).labels_

    def get_cluster_means(self):
        """Trọng tâm (mean) của từng cụm."""
        X = None  # Cần X để tính — không lưu X trong class để tiết kiệm RAM
        return None  # Xem get_cluster_means_from_X

    def get_cluster_means_from_X(self, X):
        """Trả về array (k, D) chứa trọng tâm từng cụm."""
        X = np.asarray(X, dtype=float)
        return np.array([X[c].mean(axis=0) for c in self.clusters_])

