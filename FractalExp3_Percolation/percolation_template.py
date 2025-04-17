# 学生代码模板：逾渗理论
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def generate_lattice(size=50, p=0.59, seed=None):
    """
    生成二维格点并随机占据。
    参数：
        size: 格点大小
        p: 占据概率
        seed: 随机种子
    返回：
        lattice: 0/1数组
    """
    # === 请在此处补充你的代码 ===
    pass

def find_clusters(lattice):
    """
    团簇标记算法。
    参数：
        lattice: 0/1数组
    返回：
        labeled: 标记后的数组
        n: 团簇数
    """
    # === 请在此处补充你的代码 ===
    pass

def has_percolating_cluster(labeled):
    """
    判断是否存在逾渗团簇。
    参数：
        labeled: 标记后的数组
    返回：
        percolating: 逾渗团簇标签数组
    """
    # === 请在此处补充你的代码 ===
    pass

def plot_lattice(lattice, labeled=None):
    """
    可视化格点或团簇。
    """
    # === 请在此处补充你的代码 ===
    pass

if __name__ == '__main__':
    # 可选：调用你的函数并可视化结果
    pass