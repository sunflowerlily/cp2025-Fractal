# 参考答案：扩散限制聚集（DLA）
import numpy as np
import matplotlib.pyplot as plt

def dla_simulation(size=101, n_particles=500, seed=None):
    np.random.seed(seed)
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    grid[center, center] = 1  # 种子粒子
    for _ in range(n_particles):
        x, y = np.random.choice([0, size-1]), np.random.randint(0, size)
        if np.random.rand() > 0.5:
            x, y = np.random.randint(0, size), np.random.choice([0, size-1])
        while True:
            # 随机行走
            dx, dy = np.random.choice([-1,0,1]), np.random.choice([-1,0,1])
            if dx == 0 and dy == 0:
                continue
            x = (x + dx) % size
            y = (y + dy) % size
            # 检查邻居
            neighbors = grid[max(0,x-1):x+2, max(0,y-1):y+2]
            if np.any(neighbors):
                grid[x, y] = 1
                break
            # 防止粒子走出边界
            if x <= 1 or x >= size-2 or y <= 1 or y >= size-2:
                break
    return grid

def plot_dla(grid):
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap='inferno', origin='lower')
    plt.title('DLA Cluster')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    grid = dla_simulation(size=101, n_particles=800)
    plot_dla(grid)