# 参考答案：扩散限制聚集（DLA）
import numpy as np
import matplotlib.pyplot as plt

def dla_simulation(size=101, n_particles=500, seed=None):
    np.random.seed(seed)
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    grid[center, center] = 1
    
    # 增加初始簇大小
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        grid[center+dx, center+dy] = 1
    
    for _ in range(n_particles):
        # 从更远的边界释放粒子
        x, y = np.random.choice([0, size-1]), np.random.randint(5, size-5)
        if np.random.rand() > 0.5:
            x, y = np.random.randint(5, size-5), np.random.choice([0, size-1])
            
        stuck = False
        for _ in range(size*2):  # 限制最大步数
            # 随机行走：在八邻域内随机选择一个方向
            dx, dy = np.random.choice([-1,0,1]), np.random.choice([-1,0,1])
            if dx == 0 and dy == 0:
                continue  # 保证粒子确实移动
            x = (x + dx) % size
            y = (y + dy) % size
            # 检查粒子当前位置的邻居（3x3区域）是否有已聚集粒子
            neighbors = grid[max(0,x-1):x+2, max(0,y-1):y+2]
            # 修改粘附条件，增加粘附概率
            if np.sum(neighbors) >= 1:  # 原为 np.any(neighbors)
                grid[x, y] = 1
                stuck = True
                break
                
            # 移除模运算，改为硬边界
            if x <= 0 or x >= size-1 or y <= 0 or y >= size-1:
                break
                
        if not stuck:
            continue  # 粒子未粘附则跳过
            
    return grid

def plot_dla(grid):
    """
    可视化 DLA 聚集结构。
    参数：
        grid (ndarray): DLA 结果二维数组。
    """
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap='inferno', origin='lower')  # 使用'inferno'色图增强分形结构对比
    plt.title('DLA Cluster')
    plt.axis('off')  # 隐藏坐标轴
    plt.show()

if __name__ == '__main__':
    # 运行 DLA 模型并可视化结果
    grid = dla_simulation(size=21, n_particles=1000)
    plot_dla(grid)