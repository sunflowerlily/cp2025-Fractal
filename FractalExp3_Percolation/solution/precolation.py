import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import time  # 用于计时

# --- 仿真参数 ---
L = 100  # 格点尺寸（L x L）
N_runs = 50  # 每个概率p下的模拟次数，用于计算贯穿概率
# 用于绘制贯穿概率曲线的p值（围绕理论阈值pc ~ 0.5927）
p_values_spanning = np.linspace(0.5, 0.7, 41)
p_critical_estimate = 0.5927  # 用于临界团簇可视化

# --- 辅助函数 ---

def create_lattice(L, p):
    """
    创建一个LxL的格点，每个格点以概率p被占据。
    参数:
        L (int): 格点边长
        p (float): 占据概率
    返回:
        ndarray: 0/1数组，1表示被占据，0表示空
    """
    lattice = np.random.rand(L, L) < p
    return lattice.astype(int)  # 布尔转为int（1=占据，0=空）

def find_clusters(lattice):
    """
    查找格点中的所有连通团簇，采用4邻域（冯·诺依曼邻域）。
    参数:
        lattice (ndarray): 输入格点
    返回:
        labeled_lattice (ndarray): 标记后的格点，每个团簇有唯一标签
        num_labels (int): 团簇总数（不含背景）
    """
    # 结构元素定义4邻域连通性
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labeled_lattice, num_labels = label(lattice, structure=structure)
    return labeled_lattice, num_labels

def check_spanning(labeled_lattice, L):
    """
    检查是否存在从顶行到底行贯穿的团簇。
    参数:
        labeled_lattice (ndarray): 团簇标记后的格点
        L (int): 格点边长
    返回:
        bool: 是否存在贯穿团簇
    """
    top_labels = np.unique(labeled_lattice[0, :])  # 顶部所有团簇标签
    bottom_labels = np.unique(labeled_lattice[L-1, :])  # 底部所有团簇标签
    # 找到同时出现在顶部和底部的标签（忽略背景0）
    common_labels = np.intersect1d(top_labels[top_labels > 0], bottom_labels[bottom_labels > 0])
    return len(common_labels) > 0

def get_largest_cluster_points(labeled_lattice):
    """
    获取最大团簇中所有点的坐标。
    参数:
        labeled_lattice (ndarray): 团簇标记后的格点
    返回:
        ndarray: 最大团簇的点坐标（Nx2数组）
    """
    unique_labels, counts = np.unique(labeled_lattice, return_counts=True)
    # 只存在背景或没有团簇
    if len(unique_labels) <= 1:
        return np.array([])
    # 找到最大团簇的标签（忽略背景0）
    largest_cluster_label_index = np.argmax(counts[1:]) + 1
    largest_cluster_label = unique_labels[largest_cluster_label_index]
    # 获取最大团簇所有点的坐标
    points = np.argwhere(labeled_lattice == largest_cluster_label)
    return points

def box_counting(points, L, min_box_size=1, max_box_size=None):
    """
    对点集进行盒计数法，估算分形维数。
    参数:
        points (np.array): Nx2的点坐标
        L (int): 格点边长
        min_box_size (int): 最小盒子尺寸
        max_box_size (int): 最大盒子尺寸，默认为L
    返回:
        box_sizes (np.array): 盒子尺寸数组
        box_counts (np.array): 每个尺寸下的盒子数
    """
    if points.size == 0:
        return np.array([]), np.array([])
    if max_box_size is None:
        max_box_size = L
    # 生成盒子尺寸（常用2的幂次，或对数均匀分布）
    n_scales = int(np.log2(max_box_size / min_box_size)) + 1
    box_sizes = np.logspace(np.log2(min_box_size), np.log2(max_box_size), num=n_scales, base=2.0)
    box_sizes = np.unique(np.round(box_sizes).astype(int))
    box_sizes = box_sizes[box_sizes >= min_box_size]
    box_counts = []
    actual_box_sizes = []
    for epsilon in box_sizes:
        if epsilon == 0:
            continue
        # 用集合高效存储被占据的盒子索引
        occupied_boxes = set()
        for r, c in points:
            box_r = int(r // epsilon)
            box_c = int(c // epsilon)
            occupied_boxes.add((box_r, box_c))
        if len(occupied_boxes) > 0:
            box_counts.append(len(occupied_boxes))
            actual_box_sizes.append(epsilon)
    return np.array(actual_box_sizes), np.array(box_counts)

# --- 主仿真流程 ---

if __name__ == "__main__":
    start_time = time.time()

    # 1. 计算贯穿概率
    print(f"正在计算L={L}下的贯穿概率...")
    spanning_probs = []
    for p in p_values_spanning:
        spanning_count = 0
        for run in range(N_runs):
            lattice = create_lattice(L, p)
            labeled_lattice, _ = find_clusters(lattice)
            if check_spanning(labeled_lattice, L):
                spanning_count += 1
        spanning_probs.append(spanning_count / N_runs)
        print(f"  p = {p:.4f}, 贯穿概率 = {spanning_probs[-1]:.3f}")

    spanning_probs = np.array(spanning_probs)

    # 估算pc为贯穿概率最接近0.5的位置
    pc_estimated_idx = np.argmin(np.abs(spanning_probs - 0.5))
    pc_estimated = p_values_spanning[pc_estimated_idx]
    print(f"\n估算的逾渗阈值 pc(L={L}) ≈ {pc_estimated:.4f}")

    # 2. 生成并可视化临界团簇
    print(f"\n生成接近临界点p={p_critical_estimate}的格点...")
    critical_lattice = create_lattice(L, p_critical_estimate)
    labeled_critical, num_labels = find_clusters(critical_lattice)
    print(f"共找到 {num_labels} 个团簇。")

    # 找到最大团簇
    largest_cluster_pts = get_largest_cluster_points(labeled_critical)
    print(f"最大团簇包含 {len(largest_cluster_pts)} 个点。")

    # 只显示最大团簇
    largest_cluster_viz = np.zeros((L, L), dtype=int)
    if largest_cluster_pts.size > 0:
        rows, cols = largest_cluster_pts.T
        largest_cluster_viz[rows, cols] = 1

    # 3. 对最大团簇做盒计数法
    print("\n正在对最大团簇进行盒计数法...")
    box_sizes, box_counts = box_counting(largest_cluster_pts, L)

    # 过滤掉无效数据（如计数为0或尺寸为0）
    valid_indices = (box_counts > 0) & (box_sizes > 0)
    box_sizes_valid = box_sizes[valid_indices]
    box_counts_valid = box_counts[valid_indices]

    fractal_dimension = 0.0
    if len(box_sizes_valid) > 1:
        # 计算对数值
        log_epsilon = np.log(1.0 / box_sizes_valid)  # 用1/epsilon保证斜率为正
        log_N = np.log(box_counts_valid)
        # 拟合直线，斜率即为分形维数
        coeffs = np.polyfit(log_epsilon, log_N, 1)
        fractal_dimension = coeffs[0]
        print(f"\n估算分形维数（盒计数法） D_f ≈ {fractal_dimension:.3f}")
    else:
        print("\n数据点不足，无法计算分形维数。")

    end_time = time.time()
    print(f"\n总仿真用时: {end_time - start_time:.2f} 秒")

    # --- 绘图 ---
    plt.figure(figsize=(18, 6))

    # 图1：贯穿概率曲线
    plt.subplot(1, 3, 1)
    plt.plot(p_values_spanning, spanning_probs, 'bo-', label=f'L={L}, N_runs={N_runs}')
    plt.axvline(pc_estimated, color='r', linestyle='--', label=f'估算 $p_c \\approx {pc_estimated:.4f}$')
    plt.axhline(0.5, color='grey', linestyle=':', label='概率 = 0.5')
    plt.xlabel("占据概率 (p)")
    plt.ylabel("贯穿概率 $\\Pi(p, L)$")
    plt.title("逾渗贯穿概率曲线")
    plt.legend()
    plt.grid(True)

    # 图2：最大临界团簇可视化
    plt.subplot(1, 3, 2)
    plt.imshow(largest_cluster_viz, cmap='binary', origin='lower', interpolation='nearest')
    plt.title(f"p={p_critical_estimate:.4f} (L={L})下最大团簇")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks([])
    plt.yticks([])

    # 图3：盒计数法对数-对数图
    plt.subplot(1, 3, 3)
    if len(box_sizes_valid) > 1:
        plt.plot(log_epsilon, log_N, 'gs-', label='数据点')
        plt.plot(log_epsilon, np.polyval(coeffs, log_epsilon), 'r--', label=f'拟合: $D_f \\approx {fractal_dimension:.3f}$')
        plt.xlabel("$\\log(1 / \\epsilon)$")
        plt.ylabel("$\\log(N(\\epsilon))$")
        plt.title("盒计数法对数-对数图")
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "数据不足，无法绘图", ha='center', va='center')
        plt.title("盒计数法对数-对数图")

    plt.tight_layout()
    plt.show()