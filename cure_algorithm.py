"""CURE với heap khoảng cách, PAM và DIANA; không lọc ngoại lai hai pha."""
import heapq
from numbers import Integral
import numpy as np
from scipy.spatial.distance import cdist, pdist


def validate_data(X, k=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError('Dữ liệu phải là ma trận số hữu hạn, không rỗng.')
    if k is not None and (isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= len(X)):
        raise ValueError('Số cụm k phải là số nguyên từ 1 đến số điểm.')
    return X


class KMedoids:
    """PAM: BUILD tham lam và SWAP đến cực tiểu cục bộ.

    BUILD xác định; random_state giữ tương thích API, không được sử dụng.
    """
    def __init__(self, n_clusters=2, max_iter=100, random_state=42):
        self.n_clusters, self.max_iter, self.random_state = n_clusters, max_iter, random_state

    def fit(self, X):
        X = validate_data(X, self.n_clusters)
        if not isinstance(self.max_iter, Integral) or self.max_iter < 1:
            raise ValueError('max_iter phải là số nguyên dương.')
        D = cdist(X, X)
        medoids = [int(np.argmin(D.sum(axis=0)))]
        nearest = D[:, medoids[0]].copy()
        while len(medoids) < self.n_clusters:
            costs = np.minimum(nearest[:, None], D).sum(axis=0)
            costs[medoids] = np.inf
            medoids.append(int(np.argmin(costs)))
            nearest = np.minimum(nearest, D[:, medoids[-1]])
        medoids = np.array(medoids)
        self.objective_history_ = [float(nearest.sum())]
        self.converged_ = False
        for iteration in range(self.max_iter):
            dm = D[:, medoids]
            owner, nearest = dm.argmin(axis=1), dm.min(axis=1)
            second = (np.partition(dm, 1, axis=1)[:, 1] if self.n_clusters > 1
                      else np.full(len(X), np.inf))
            best_cost, swap = float(nearest.sum()), None
            for slot in range(self.n_clusters):
                without = np.where(owner == slot, second, nearest)
                costs = np.minimum(without[:, None], D).sum(axis=0)
                costs[medoids] = np.inf
                candidate = int(np.argmin(costs))
                if costs[candidate] < best_cost - 1e-10:
                    best_cost, swap = float(costs[candidate]), (slot, candidate)
            if swap is None:
                self.converged_ = True
                break
            medoids[swap[0]] = swap[1]
            self.objective_history_.append(best_cost)
        self.n_iter_ = iteration + 1
        self.medoid_indices_ = medoids
        self.labels_ = D[:, medoids].argmin(axis=1)
        # Gỡ hòa khi điểm trùng nhau: medoid sở hữu chính nó, không đổi chi phí.
        self.labels_[medoids] = np.arange(self.n_clusters)
        self.cluster_centers_ = X[medoids].copy()
        self.inertia_ = float(D[np.arange(len(X)), medoids[self.labels_]].sum())
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


class CURECluster:
    def __init__(self, points, cluster_id, indices=None):
        self.cluster_id = cluster_id
        self.points = np.atleast_2d(np.asarray(points, dtype=float))
        self.indices = np.asarray(indices if indices is not None else np.arange(len(self.points)), dtype=int)
        self.mean = self.points.mean(axis=0)
        self.rep_points = self.points.copy()

    def update_representatives(self, c, alpha):
        self.mean = self.points.mean(axis=0)
        if len(self.points) <= c:
            selected = self.points.copy()
        else:
            first = int(np.argmax(np.linalg.norm(self.points - self.mean, axis=1)))
            indices = [first]
            nearest = np.linalg.norm(self.points - self.points[first], axis=1)
            for _ in range(1, c):
                nearest[indices] = -np.inf
                nxt = int(np.argmax(nearest))
                indices.append(nxt)
                nearest = np.minimum(nearest, np.linalg.norm(self.points - self.points[nxt], axis=1))
            selected = self.points[indices]
        self.rep_points = selected + alpha * (self.mean - selected)


class CURE:
    """CURE dùng heap với vô hiệu hóa cặp cũ theo phiên bản cụm.

    Với c và số chiều cố định: O(s² log s) thời gian, O(s²) bộ nhớ.
    Giữ nhãn mẫu đã gom; chỉ predict cho điểm ngoài mẫu. history_ ghi phép gom
    và đại diện để giao diện phát lại. Chưa có phân hoạch/lọc ngoại lai hai pha.
    """
    def __init__(self, n_clusters=2, n_representatives=5, shrink_factor=0.5,
                 sample_size=None, random_state=42, record_history=True):
        self.n_clusters, self.n_representatives = n_clusters, n_representatives
        self.shrink_factor, self.sample_size = shrink_factor, sample_size
        self.random_state, self.record_history = random_state, record_history
        self.clusters_, self.history_, self.labels_ = [], [], None

    def _cluster_dist(self, c1, c2):
        return float(cdist(c1.rep_points, c2.rep_points).min())

    def fit(self, X):
        X = validate_data(X, self.n_clusters)
        if not isinstance(self.n_representatives, Integral) or self.n_representatives < 1:
            raise ValueError('Số điểm đại diện c phải là số nguyên dương.')
        if not np.isfinite(self.shrink_factor) or not 0 <= self.shrink_factor <= 1:
            raise ValueError('Hệ số co alpha phải thuộc [0, 1].')
        if self.sample_size is not None and (
            not isinstance(self.sample_size, Integral) or self.sample_size < self.n_clusters
        ):
            raise ValueError('Kích thước mẫu phải là số nguyên không nhỏ hơn k.')
        self.n_features_in_ = X.shape[1]
        self.history_ = []
        self.sample_indices_ = np.arange(len(X))
        if self.sample_size is not None and self.sample_size < len(X):
            self.sample_indices_ = np.random.RandomState(self.random_state).choice(
                len(X), self.sample_size, replace=False)
        sample = X[self.sample_indices_]
        clusters = {i: CURECluster([p], i, [self.sample_indices_[i]]) for i, p in enumerate(sample)}
        versions = {i: 0 for i in clusters}
        D = cdist(sample, sample)
        heap = [(float(D[i, j]), i, j, 0, 0) for i in clusters for j in range(i + 1, len(sample))]
        heapq.heapify(heap)
        del D
        while len(clusters) > self.n_clusters:
            distance, u, v, vu, vv = heapq.heappop(heap)
            if u not in clusters or v not in clusters or versions[u] != vu or versions[v] != vv:
                continue
            left, right = clusters[u], clusters[v]
            merged = CURECluster(np.vstack((left.points, right.points)), u,
                                 np.concatenate((left.indices, right.indices)))
            merged.update_representatives(self.n_representatives, self.shrink_factor)
            clusters[u] = merged
            del clusters[v]
            versions[u] += 1
            if self.record_history:
                self.history_.append({'left': u, 'right': v, 'distance': distance,
                                      'size': len(merged.points), 'mean': merged.mean.tolist(),
                                      'representatives': merged.rep_points.tolist()})
            for w in clusters:
                if w != u:
                    a, b = sorted((u, w))
                    heapq.heappush(heap, (self._cluster_dist(merged, clusters[w]), a, b,
                                          versions[a], versions[b]))
        self.clusters_ = [clusters[i] for i in sorted(clusters)]
        self.labels_ = np.empty(len(X), dtype=int)
        for label, cluster in enumerate(self.clusters_):
            self.labels_[cluster.indices] = label
        outside = np.ones(len(X), dtype=bool)
        outside[self.sample_indices_] = False
        if outside.any():
            self.labels_[outside] = self.predict(X[outside])
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def predict(self, X):
        if not self.clusters_:
            raise ValueError('Cần fit CURE trước khi predict.')
        X = validate_data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Số thuộc tính không khớp dữ liệu đã fit.')
        reps = np.vstack(self.get_representatives())
        owners = np.concatenate([np.full(len(c.rep_points), i) for i, c in enumerate(self.clusters_)])
        return np.concatenate([owners[cdist(batch, reps).argmin(axis=1)]
                               for batch in np.array_split(X, max(1, (len(X) + 4095) // 4096))])

    def get_representatives(self):
        return [c.rep_points.copy() for c in self.clusters_]

    def get_cluster_means(self):
        return np.array([c.mean for c in self.clusters_])


class DIANA:
    """DIANA chính xác: tách cụm có đường kính lớn nhất; chuyển từng điểm.

    max_points_full giữ tương thích API, đường kính luôn được tính chính xác.
    """
    def __init__(self, n_clusters=2, max_points_full=None):
        self.n_clusters, self.max_points_full = n_clusters, max_points_full

    def _diameter(self, X, indices):
        return float(pdist(X[indices]).max()) if len(indices) > 1 else 0.0

    def _split_distances(self, distances, indices):
        n = len(indices)
        if n < 2:
            return list(indices), []
        D = distances[np.ix_(indices, indices)]
        splinter = int(np.argmax(D.sum(axis=1)))
        main = np.ones(n, dtype=bool)
        main[splinter] = False
        split = ~main
        sum_main, sum_split = D[:, main].sum(axis=1), D[:, split].sum(axis=1)
        while main.sum() > 1:
            candidates = np.flatnonzero(main)
            gains = sum_main[candidates] / (main.sum() - 1) - sum_split[candidates] / split.sum()
            best = int(np.argmax(gains))
            if gains[best] <= 0:
                break
            move = candidates[best]
            main[move], split[move] = False, True
            sum_main -= D[:, move]
            sum_split += D[:, move]
        return np.asarray(indices)[main].tolist(), np.asarray(indices)[split].tolist()

    def _split_cluster(self, X, indices):
        return self._split_distances(cdist(X, X), indices)

    def fit(self, X):
        X = validate_data(X, self.n_clusters)
        D = cdist(X, X)
        clusters, diameters = [list(range(len(X)))], [float(D.max())]
        while len(clusters) < self.n_clusters:
            # Bỏ singleton ngay cả khi tất cả tọa độ trùng nhau.
            eligible = [i for i, cluster in enumerate(clusters) if len(cluster) > 1]
            idx = max(eligible, key=lambda i: diameters[i])
            main, split = self._split_distances(D, clusters.pop(idx))
            diameters.pop(idx)
            for part in (main, split):
                clusters.append(part)
                diameters.append(float(D[np.ix_(part, part)].max()))
        self.labels_ = np.empty(len(X), dtype=int)
        for label, cluster in enumerate(clusters):
            self.labels_[cluster] = label
        self.clusters_ = clusters
        self.cluster_centers_ = np.array([X[c].mean(axis=0) for c in clusters])
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def get_cluster_means(self):
        return self.cluster_centers_.copy()

    def get_cluster_means_from_X(self, X):
        return np.array([np.asarray(X)[c].mean(axis=0) for c in self.clusters_])
