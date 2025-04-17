# 参考答案：曼德博集合与朱利亚集合
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(width=800, height=600, max_iter=100, xlim=(-2,1), ylim=(-1.5,1.5)):
    x = np.linspace(xlim[0], xlim[1], width)
    y = np.linspace(ylim[0], ylim[1], height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    img = np.zeros(C.shape, dtype=int)
    for i in range(max_iter):
        mask = np.abs(Z) < 2
        Z[mask] = Z[mask]**2 + C[mask]
        img[mask & (np.abs(Z) >= 2)] = i
    return img

def plot_mandelbrot():
    img = mandelbrot()
    plt.imshow(img, cmap='hot', extent=(-2,1,-1.5,1.5))
    plt.title('Mandelbrot Set')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.colorbar(label='Escape Time')
    plt.show()

def julia(c, width=800, height=600, max_iter=100, xlim=(-2,2), ylim=(-2,2)):
    x = np.linspace(xlim[0], xlim[1], width)
    y = np.linspace(ylim[0], ylim[1], height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    img = np.zeros(Z.shape, dtype=int)
    for i in range(max_iter):
        mask = np.abs(Z) < 2
        Z[mask] = Z[mask]**2 + c
        img[mask & (np.abs(Z) >= 2)] = i
    return img

def plot_julia(c):
    img = julia(c)
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