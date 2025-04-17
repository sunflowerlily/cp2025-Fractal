# 参考答案：分形维数计算（盒子计数法）
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk

def box_counting(data, epsilons):
    N = []
    for eps in epsilons:
        size = int(np.ceil(data.max() / eps))
        grid = np.zeros((size, size), dtype=bool)
        for x, y in data:
            i = int(x // eps)
            j = int(y // eps)
            grid[i, j] = True
        N.append(np.sum(grid))
    return np.array(N)

def generate_sierpinski_points(level=5):
    points = [(0,0), (1,0), (0.5,np.sqrt(3)/2)]
    pts = set(points)
    for _ in range(level):
        new_pts = set()
        for (x1,y1) in pts:
            for (x2,y2) in points:
                new_pts.add((x1/2+x2/2, y1/2+y2/2))
        pts = new_pts
    arr = np.array(list(pts))
    arr -= arr.min(axis=0)
    arr /= arr.max(axis=0)
    arr *= 100
    return arr

def plot_box_counting(data, epsilons, N):
    plt.figure()
    plt.plot(np.log(1/epsilons), np.log(N), 'o-')
    plt.xlabel('log(1/epsilon)')
    plt.ylabel('log(N(epsilon))')
    plt.title('Box-Counting Dimension')
    plt.show()

if __name__ == '__main__':
    data = generate_sierpinski_points(level=5)
    epsilons = np.array([32, 16, 8, 4, 2, 1])
    N = box_counting(data, epsilons)
    plot_box_counting(data, epsilons, N)