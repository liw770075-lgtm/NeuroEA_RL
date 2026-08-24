import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time
from scipy import stats  # 导入 scipy 用于统计计算


# --- 修改后的测试用 ELA 计算函数 ---
def _adjusted_r2(r2, n, p):
    if p >= n:
        return 0.0
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def compute_ela_features(X, Y, seed=None):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).flatten()
    n, d = X.shape

    if n <= 1 or d == 0:
        return np.zeros(9)

    # --- 1. Meta-model features ---
    # lin_simple.intercept
    lr1 = LinearRegression(fit_intercept=True).fit(X, Y)
    intercept = lr1.intercept_

    # quad_simple.adj_r2
    X_quad = np.hstack([X, X ** 2])
    lr2 = LinearRegression(fit_intercept=True).fit(X_quad, Y)
    r2_quad = r2_score(Y, lr2.predict(X_quad))
    adj_r2_quad = _adjusted_r2(r2_quad, n, 2 * d)

    # lin_w_interact.adj_r2 (limit to first min(d,5) dims to avoid explosion)
    d_use = min(d, 5)
    X_sub = X[:, :d_use]
    interactions = []
    for i in range(d_use):
        for j in range(i + 1, d_use):
            interactions.append(X_sub[:, i] * X_sub[:, j])
    if interactions:
        X_inter = np.hstack([X, np.column_stack(interactions)])
        lr3 = LinearRegression(fit_intercept=True).fit(X_inter, Y)
        r2_inter = r2_score(Y, lr3.predict(X_inter))
        p_total = d + len(interactions)
        adj_r2_inter = _adjusted_r2(r2_inter, n, p_total)
    else:
        adj_r2_inter = 0.0

    # --- 2. Information Content ---
    eps = 0.05 * (Y.max() - Y.min() + 1e-12)
    diff = np.diff(Y)
    signs = np.sign(diff)
    signs[signs == 0] = 1
    m0 = np.mean(signs[:-1] != signs[1:]) if len(signs) > 1 else 0.0

    uniq, cnt = np.unique(signs, return_counts=True)
    prob = cnt / cnt.sum()
    h_max = -np.sum(prob * np.log(prob + 1e-12))

    pairwise = np.abs(Y[:, None] - Y[None, :])
    triu_vals = pairwise[np.triu_indices(n, k=1)]
    eps_ratio = np.mean(triu_vals < eps) if triu_vals.size > 0 else 0.0

    # --- 3. NBC ---
    dists = squareform(pdist(X))
    np.fill_diagonal(dists, np.inf)
    k = min(5, n - 1)
    neighbor_idx = np.argpartition(dists, k, axis=1)[:, :k]

    better_ratios = []
    dist_ratios = []
    for i in range(n):
        nb = neighbor_idx[i]
        better_count = np.sum(Y[nb] < Y[i])
        better_ratios.append(better_count / k)

        better_mask = Y[nb] < Y[i]
        if np.any(better_mask):
            min_better = dists[i, nb][better_mask].min()
        else:
            min_better = dists[i].min()
        min_nb_dist = dists[i, nb].min()
        dist_ratios.append(min_better / (min_nb_dist + 1e-12))

    mean_ratio = np.mean(better_ratios)
    cv = np.std(dist_ratios) / (np.mean(dist_ratios) + 1e-12)

    # --- 4. Peaks ---
    Y_norm = (Y - Y.min()) / (Y.max() - Y.min() + 1e-12)
    n_peaks = np.sum(Y_norm < 0.1)

    # --- Assemble & sanitize ---
    feats = [
        intercept,
        adj_r2_quad,
        adj_r2_inter,
        m0,
        h_max,
        eps_ratio,
        mean_ratio,
        cv,
        n_peaks
    ]
    feats = [float(f) for f in feats]
    feats = [0.0 if np.isnan(x) or x < 0 else min(x, 100.0) for x in feats]
    return np.array(feats[:9])
# def compute_ela_features(X, Y, seed=42):
#     """
#     修正版：使用 scipy 计算基础统计特征作为 ELA 占位符
#     """
#     # 基础统计特征
#     f_mean = np.mean(Y)
#     f_std = np.std(Y)
#     f_range = np.max(Y) - np.min(Y)
#
#     # 使用 scipy 计算偏度和峰度 (针对 Y 或 X 的均值)
#     y_skew = stats.skew(Y)
#     y_kurt = stats.kurtosis(Y)
#
#     return np.array([f_mean, f_std, f_range, y_skew, y_kurt])


# --- 你的主提取逻辑 (保持不变，仅确认属性名) ---
def get_ela_from_neuroea_batch(problem_names, D=10, ela_sample=50, seed=42):
    """
    修改版：处理问题名称列表，返回 ELA 特征列表
    """
    ela_list = []
    for name in problem_names:
        print(f"[*] 正在为 {name} 提取 ELA 特征...")
        N = ela_sample * D
        para = [N, 1, D, 10000]

        try:
            from NeuroEA_GEA.Problems import get_problem_instance
            # 实例化当前循环中的问题
            problem = get_problem_instance(name, *para)
            initial_solutions = problem.initialization()

            # 提取数据
            X_np = np.array([sol.dec for sol in initial_solutions])
            Y_np = np.array([sol.obj for sol in initial_solutions]).flatten()

            # 调用你之前的 9 维计算函数
            ela_vec = compute_ela_features(X_np, Y_np, seed=seed)
            ela_list.append(ela_vec)
        except Exception as e:
            print(f"[错误] {name} 计算失败: {e}")
            # 如果失败，填充 9 维零向量保持维度一致
            ela_list.append(np.zeros(9))

    return np.array(ela_list)


if __name__ == "__main__":
    # 再次运行测试
    for p_name in ['BBOB_F1', 'BBOB_F2']:
        print(f"[*] 测试 {p_name}...")
        features = get_ela_from_neuroea(p_name)
        if features is not None:
            print(f"[成功] 特征向量: {features.round(4)}")