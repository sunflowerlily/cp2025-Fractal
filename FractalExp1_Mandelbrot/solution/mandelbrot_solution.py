# 参考答案：曼德博集合与朱利亚集合
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(width=800, height=600, max_iter=100, xlim=(-2,1), ylim=(-1.5,1.5)):
    # 生成横坐标和纵坐标的等间距数值
    x = np.linspace(xlim[0], xlim[1], width)  # 横坐标范围
    y = np.linspace(ylim[0], ylim[1], height) # 纵坐标范围
    # 生成复平面上的网格点
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y  # 构造复数平面上的每个点
    Z = np.zeros_like(C)  # 初始化Z为0，形状与C相同
    img = np.zeros(C.shape, dtype=int)  # 用于记录每个点的逃逸时间
    for i in range(max_iter):
        # 创建一个布尔掩码，标记所有模小于2的复数点（未逃逸的点）
        mask = np.abs(Z) < 2
        # 对于未逃逸的点，进行迭代计算 Z = Z^2 + C
        Z[mask] = Z[mask]**2 + C[mask]
        # 记录本次迭代中新逃逸的点（模刚好大于等于2的点）的逃逸时间
        img[mask & (np.abs(Z) >= 2)] = i
    return img

def plot_mandelbrot():
    # 生成曼德博集合图像
    img = mandelbrot()
    # 显示图像
    plt.imshow(img, cmap='hot', extent=(-2,1,-1.5,1.5))
    plt.title('Mandelbrot Set')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.colorbar(label='Escape Time')
    plt.show()

def julia(c, width=800, height=600, max_iter=100, xlim=(-2,2), ylim=(-2,2)):
    # 生成横坐标和纵坐标的等间距数值
    x = np.linspace(xlim[0], xlim[1], width)
    y = np.linspace(ylim[0], ylim[1], height)
    # 生成复平面上的网格点
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y  # 初始化Z为复平面上的每个点
    img = np.zeros(Z.shape, dtype=int)  # 用于记录每个点的逃逸时间
    for i in range(max_iter):
        # 创建掩码，标记未逃逸的点
        mask = np.abs(Z) < 2
        # 对未逃逸的点进行迭代 Z = Z^2 + c
        Z[mask] = Z[mask]**2 + c
        # 记录本次迭代中新逃逸的点的逃逸时间
        img[mask & (np.abs(Z) >= 2)] = i
    return img

def plot_julia(c):
    # 生成朱利亚集合图像
    img = julia(c)
    # 显示图像
    plt.imshow(img, cmap='hot', extent=(-2,2,-2,2))
    plt.title(f'Julia Set for c={c}')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.colorbar(label='Escape Time')
    plt.show()

if __name__ == '__main__':
    plot_mandelbrot()
    # 示例：绘制c=0.355+0.355j的Julia集合
    plot_julia(0.355 + 0.355j)