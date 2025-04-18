import numpy as np
import matplotlib.pyplot as plt
import math

# --- Koch Curve Generation ---

def koch_curve_points(p1, p2, level):
    """
    递归生成Koch曲线某一线段的所有点。
    参数：
        p1 (np.array): 起点 [x, y]。
        p2 (np.array): 终点 [x, y]。
        level (int): 递归层数。
    返回：
        list: 当前层级曲线的点（np.array），包含p1但不包含p2（拼接时避免重复）。
    """
    if level == 0:
        return [p1]
    else:
        # 计算分割点
        v = (p2 - p1) / 3.0
        p_a = p1 + v
        p_b = p1 + 2.0 * v
        print(v,p1,p2)
        # 计算三角形顶点
        angle = np.pi / 3.0  # 60度
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        v_rotated = np.dot(rotation_matrix, v)
        p_tip = p_a + v_rotated

        # 递归生成四段
        points = []
        points.extend(koch_curve_points(p1, p_a, level - 1))
        points.extend(koch_curve_points(p_a, p_tip, level - 1))
        points.extend(koch_curve_points(p_tip, p_b, level - 1))
        points.extend(koch_curve_points(p_b, p2, level - 1))
        return points

def box_counting(points, n_scales=10):
    """
    对二维点集进行盒计数法，估算分形维数。
    参数：
        points (np.array): Nx2的[x, y]坐标数组。
        n_scales (int): 盒子尺寸（epsilon）的数量，对数分布。
    返回：
        box_sizes (np.array): 使用的盒子尺寸（epsilon）。
        box_counts (np.array): 每个尺寸下的盒子数N(epsilon)。
    """
    if points.shape[0] < 2:  # 至少需要两个点
        return np.array([]), np.array([])

    # 计算点集范围
    min_coords = points.min(axis=0)  # 计算点集在每个维度上的最小坐标值
    max_coords = points.max(axis=0)  # 计算点集在每个维度上的最大坐标值
    range_coords = max_coords - min_coords  # 计算点集在每个维度上的范围（最大值减最小值）

    # 防止所有点重合导致范围为0
    if np.all(range_coords == 0):
        max_dim_size = 1.0
    else:
        max_dim_size = max(range_coords[range_coords > 0], default=1.0)

    min_box_size = max_dim_size / (2**n_scales)
    if min_box_size == 0:
        min_box_size = np.finfo(float).eps

    max_box_size = max_dim_size

    # 生成对数分布的盒子尺寸
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=n_scales)
    # 去除重复的盒子尺寸，因为logspace可能会生成非常接近的值
    box_sizes = np.unique(box_sizes)
    box_sizes = box_sizes[box_sizes > 0]

    # 初始化两个空列表
    # box_counts用于存储每个尺度下的非空盒子数量
    box_counts = []
    # actual_box_sizes用于存储实际使用的盒子尺寸
    actual_box_sizes = []

    for epsilon in box_sizes:
        # 用集合存储被占据的盒子索引
        occupied_boxes = set()
        for p in points:
            box_index = tuple(np.floor((p - min_coords) / epsilon).astype(int))
            occupied_boxes.add(box_index)

        count = len(occupied_boxes)
        if count > 0:
            box_counts.append(count)
            actual_box_sizes.append(epsilon)

    return np.array(actual_box_sizes), np.array(box_counts)

if __name__ == "__main__":
    # 参数设置
    recursion_level = 5  # Koch曲线递归层数
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])

    # 1. 生成Koch曲线点
    print(f"生成Koch曲线（递归层数 {recursion_level}）的点...")
    koch_points_list = koch_curve_points(p1, p2, recursion_level)
    koch_points_list.append(p2)  # 补上终点
    koch_points = np.array(koch_points_list)
    print(f"共生成 {len(koch_points)} 个点。")

    # 2. 盒计数法
    print("进行盒计数法计算...")
    box_sizes, box_counts = box_counting(koch_points, n_scales=5)  # 使用更多尺度

    # 过滤无效数据
    # 筛选有效的数据点：盒子数量和盒子尺寸都必须大于0
    valid_indices = (box_counts > 0) & (box_sizes > 0)
    box_sizes_valid = box_sizes[valid_indices]
    box_counts_valid = box_counts[valid_indices]

    fractal_dimension = 0.0
    coeffs = []
    if len(box_sizes_valid) > 1:
        # 计算对数值：log(N) vs log(1/epsilon)
        log_inv_epsilon = np.log10(1.0 / box_sizes_valid)
        log_N = np.log10(box_counts_valid)

        # 拟合直线，斜率即为分形维数
        coeffs = np.polyfit(log_inv_epsilon, log_N, 1)
        fractal_dimension = coeffs[0]
        print(f"\n估算分形维数（盒计数法） D_f ≈ {fractal_dimension:.4f}")
    else:
        print("\n有效数据点不足，无法计算分形维数。")

    # 理论分形维数
    theoretical_dim = np.log(4) / np.log(3)
    print(f"理论分形维数 D_H = log(4)/log(3) ≈ {theoretical_dim:.4f}")

    # --- 绘图 ---
    plt.figure(figsize=(12, 6))

    # 图1：Koch曲线可视化
    plt.subplot(1, 2, 1)
    plt.plot(koch_points[:, 0], koch_points[:, 1], 'b-', linewidth=0.8)
    plt.title(f"Koch Curve (Recursion Level {recursion_level})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')  # 保持比例
    plt.grid(True, linestyle='--', alpha=0.6)

    # 图2：盒计数法对数-对数图
    plt.subplot(1, 2, 2)
    if len(box_sizes_valid) > 1:
        plt.plot(log_inv_epsilon, log_N, 'gs', markersize=8, label='Data points')
        # 拟合直线
        plt.plot(log_inv_epsilon, np.polyval(coeffs, log_inv_epsilon), 'r--',
                 label=f'Fit: $D_f \\approx {fractal_dimension:.4f}$')
        plt.xlabel("$\\log_{10}(1 / \\epsilon)$")
        plt.ylabel("$\\log_{10}(N(\\epsilon))$")
        plt.title("Box Counting Log-Log Plot")
        # 理论斜率对比
        C_theory = np.mean(log_N) - theoretical_dim * np.mean(log_inv_epsilon)
        plt.plot(log_inv_epsilon, theoretical_dim * log_inv_epsilon + C_theory, 'k:', alpha=0.7,
                 label=f'Theory: $D_f \\approx {theoretical_dim:.4f}$')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "Not enough data for plot", ha='center', va='center')
        plt.title("Box Counting Log-Log Plot")

    plt.tight_layout()
    plt.show()