# 参考答案：逾渗理论
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def generate_lattice(size=50, p=0.59, seed=None):
    """
    生成一个二维正方形格点（lattice），每个格点以概率p被占据（1），否则为空（0）。
    参数:
        size (int): 格点的边长（生成 size x size 的格点）。
        p (float): 占据概率。
        seed (int or None): 随机种子，保证实验可复现。
    返回:
        ndarray: 占据情况的二维数组（1为占据，0为未占据）。
    """
    np.random.seed(seed)
    return (np.random.rand(size, size) < p).astype(int)

def find_clusters(lattice):
    """
    对格点进行团簇标记，识别所有连通的团簇。
    参数:
        lattice (ndarray): 输入的二维格点数组。
    返回:
        labeled (ndarray): 标记后的二维数组，每个团簇有唯一标签。
        n (int): 团簇总数。
    """
    # 结构元素定义上下左右为连通（4邻域）
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled, n = label(lattice, structure)
    return labeled, n

def has_percolating_cluster(labeled):
    """
    判断是否存在逾渗团簇（即有团簇从顶端连到底端）。
    参数:
        labeled (ndarray): 团簇标记后的二维数组。
    返回:
        ndarray: 逾渗团簇的标签（可能有多个）。
    """
    top = np.unique(labeled[0, :])      # 顶部所有团簇标签
    bottom = np.unique(labeled[-1, :])  # 底部所有团簇标签
    percolating = np.intersect1d(top, bottom)  # 同时出现在顶和底的团簇
    return percolating[percolating != 0]       # 排除背景（0）

def plot_lattice(lattice, labeled=None):
    """
    可视化格点或团簇分布。
    参数:
        lattice (ndarray): 原始格点数组。
        labeled (ndarray or None): 团簇标记数组，若为None则只显示占据情况。
    """
    plt.figure(figsize=(6,6))
    if labeled is None:
        plt.imshow(lattice, cmap='gray_r', origin='lower')
    else:
        plt.imshow(labeled, cmap='nipy_spectral', origin='lower')
    plt.title('Percolation Lattice')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # 设置格点大小和占据概率
    size = 50
    p = 0.59
    # 生成格点
    lattice = generate_lattice(size, p, seed=42)
    # 团簇标记
    labeled, n = find_clusters(lattice)
    # 检查是否存在逾渗团簇
    perc = has_percolating_cluster(labeled)
    print(f'Percolating cluster labels: {perc}')
    # 可视化
    plot_lattice(lattice, labeled)