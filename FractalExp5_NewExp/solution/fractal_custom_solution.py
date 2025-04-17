# 参考答案：自定义分形实验（以Koch雪花为例）
import numpy as np
import matplotlib.pyplot as plt

def koch_snowflake(order=3, scale=10):
    def _koch_snowflake_complex(order):
        if order == 0:
            angles = np.array([0, 2*np.pi/3, 4*np.pi/3, 0])
            return scale * np.exp(1j * angles)
        else:
            Z = _koch_snowflake_complex(order-1)
            Z_new = []
            for i in range(len(Z)-1):
                z0, z1 = Z[i], Z[i+1]
                dz = z1 - z0
                Z_new += [z0,
                          z0 + dz/3,
                          z0 + dz/3 + (dz/3)*np.exp(1j*np.pi/3),
                          z0 + dz*2/3]
            Z_new.append(Z[-1])
            return np.array(Z_new)
    points = _koch_snowflake_complex(order)
    return np.column_stack((points.real, points.imag))

def generate_custom_fractal(params=None):
    order = params["order"] if params and "order" in params else 3
    scale = params["scale"] if params and "scale" in params else 10
    return koch_snowflake(order, scale)

def plot_fractal(points):
    plt.figure(figsize=(6,6))
    plt.plot(points[:,0], points[:,1], 'b-')
    plt.axis('equal')
    plt.title('Koch Snowflake')
    plt.show()

if __name__ == '__main__':
    points = generate_custom_fractal({"order":4, "scale":10})
    plot_fractal(points)