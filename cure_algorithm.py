"""
Thuật toán phân cụm CURE (Clustering Using REpresentatives)
Cài đặt tối ưu hóa cho môn Khai phá dữ liệu - ĐH Công Thương TP.HCM (HUIT)
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

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
        X = np.asarray(X, dtype=float)
        n_samples = len(X)
        
        # 1. Lấy mẫu ngẫu nhiên nếu kích thước dữ liệu lớn
        if self.sample_size is not None and self.sample_size < n_samples:
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(n_samples, size=self.sample_size, replace=False)
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
        from scipy.spatial.distance import pdist, squareform
        d_condensed = pdist(X_sample)
        dist_matrix = squareform(d_condensed)
        np.fill_diagonal(dist_matrix, np.inf)
        
        active_ids = list(range(N))
        
        # 2. Gom cụm phân cấp tối ưu (duy trì ma trận khoảng cách)
        while len(active_ids) > self.n_clusters:
            # Tìm cặp cụm (u, v) có khoảng cách nhỏ nhất trong active_ids
            # Sub-matrix của active_ids
            idx_map = {idx: i for i, idx in enumerate(active_ids)}
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
