# 参考答案：逾渗理论
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def generate_lattice(size=50, p=0.59, seed=None):
    np.random.seed(seed)
    return (np.random.rand(size, size) < p).astype(int)

def find_clusters(lattice):
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled, n = label(lattice, structure)
    return labeled, n

def has_percolating_cluster(labeled):
    top = np.unique(labeled[0, :])
    bottom = np.unique(labeled[-1, :])
    percolating = np.intersect1d(top, bottom)
    return percolating[percolating != 0]

def plot_lattice(lattice, labeled=None):
    plt.figure(figsize=(6,6))
    if labeled is None:
        plt.imshow(lattice, cmap='gray_r', origin='lower')
    else:
        plt.imshow(labeled, cmap='nipy_spectral', origin='lower')
    plt.title('Percolation Lattice')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    size = 50
    p = 0.59
    lattice = generate_lattice(size, p, seed=42)
    labeled, n = find_clusters(lattice)
    perc = has_percolating_cluster(labeled)
    print(f'Percolating cluster labels: {perc}')
    plot_lattice(lattice, labeled)