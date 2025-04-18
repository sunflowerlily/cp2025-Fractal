import numpy as np
import matplotlib.pyplot as plt
import random

# --- Sierpinski Triangle Generation (Chaos Game) ---

def sierpinski_chaos_game(n_points, vertices):
    """
    使用混沌游戏算法生成谢尔宾斯基三角形的点集。
    参数：
        n_points (int): 要生成的点数（不包括初始烧入期）。
        vertices (np.array): 初始三角形三个顶点的坐标，形状为3x2的numpy数组。
    返回：
        np.array: 近似谢尔宾斯基三角形的(n_points)x2的点集数组。
    """
    if vertices.shape != (3, 2):
        raise ValueError("Vertices must be a 3x2 numpy array.")

    points = np.zeros((n_points, 2))
    # Start at one of the vertices (or a random point inside)
    current_point = vertices[0]

    # Burn-in period (optional, but good practice)
    burn_in = 100
    for _ in range(burn_in):
        # Choose a random vertex index (0, 1, or 2)
        vertex_index = random.randint(0, 2)
        chosen_vertex = vertices[vertex_index]
        # Move halfway towards the chosen vertex
        current_point = (current_point + chosen_vertex) / 2.0

    # Generate the points for the dataset
    for i in range(n_points):
        vertex_index = random.randint(0, 2)
        chosen_vertex = vertices[vertex_index]
        current_point = (current_point + chosen_vertex) / 2.0
        points[i] = current_point

    return points

# --- Box Counting Function (same as before) ---

def box_counting(points, n_scales=15):
    """
    对二维点集进行盒计数法，估算分形维数。
    参数：
        points (np.array): Nx2的[x, y]坐标数组。
        n_scales (int): 盒子尺寸（epsilon）的数量，对数分布。
    返回：
        box_sizes (np.array): 使用的盒子尺寸（epsilon）。
        box_counts (np.array): 每个尺寸下的盒子数N(epsilon)。
    """
    if points.shape[0] < 2:
        return np.array([]), np.array([])

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    range_coords = max_coords - min_coords

    if np.all(range_coords == 0):
         max_dim_size = 1.0
    else:
        # Use max range, default 1 if range is 0 in one dim but not others
        max_dim_size = max(range_coords[range_coords > 0], default=1.0)

    # Ensure min_box_size is reasonable, avoid zero
    # Smallest box relates to the number of points & scales
    # Heuristic: aim for smallest box to be smaller than typical point spacing
    min_box_size = max_dim_size / (2**n_scales)
    if min_box_size == 0:
        min_box_size = np.finfo(float).eps

    max_box_size = max_dim_size

    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=n_scales)
    box_sizes = np.unique(box_sizes)
    box_sizes = box_sizes[box_sizes > 0]

    box_counts = []
    actual_box_sizes = []

    for epsilon in box_sizes:
        occupied_boxes = set()
        # Calculate box indices relative to the minimum coordinate
        indices = np.floor((points - min_coords) / epsilon).astype(int)
        # Add indices tuples to the set
        occupied_boxes.update(map(tuple, indices))

        count = len(occupied_boxes)
        if count > 0:
            box_counts.append(count)
            actual_box_sizes.append(epsilon)

    return np.array(actual_box_sizes), np.array(box_counts)

# --- Main Simulation and Plotting ---

if __name__ == "__main__":
    # Parameters
    num_points = 200000  # Number of points for the chaos game
    # Define vertices of an equilateral triangle (or any triangle)
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2.0]
    ])

    # 1. Generate Sierpinski Triangle Points
    print(f"Generating {num_points} Sierpinski triangle points using Chaos Game...")
    sierpinski_points = sierpinski_chaos_game(num_points, vertices)
    print("Point generation complete.")

    # 2. Perform Box Counting
    print("Performing box counting...")
    box_sizes, box_counts = box_counting(sierpinski_points, n_scales=8) # Use more scales

    # Filter out potential issues for log-log plot
    valid_indices = (box_counts > 1) & (box_sizes > 0) # Need count > 1 for log
    box_sizes_valid = box_sizes[valid_indices]
    box_counts_valid = box_counts[valid_indices]

    fractal_dimension = 0.0
    coeffs = []
    if len(box_sizes_valid) > 1:
        # Calculate log values: log(N) vs log(1/epsilon) (using base 10)
        log_inv_epsilon = np.log10(1.0 / box_sizes_valid)
        log_N = np.log10(box_counts_valid)

        # Fit a line (degree 1 polynomial) to the log-log data
        # Select a good range for fitting (avoid smallest/largest scales if noisy)
        # Heuristic: use middle 60-80% of points for fitting
        fit_start_index = int(len(log_inv_epsilon) * 0.1)
        fit_end_index = int(len(log_inv_epsilon) * 0.9)
        if fit_end_index - fit_start_index < 2: # Ensure enough points for fit
            fit_start_index = 0
            fit_end_index = len(log_inv_epsilon)
            
        log_inv_epsilon_fit = log_inv_epsilon[fit_start_index:fit_end_index]
        log_N_fit = log_N[fit_start_index:fit_end_index]

        if len(log_inv_epsilon_fit) >= 2:
             coeffs = np.polyfit(log_inv_epsilon_fit, log_N_fit, 1)
             fractal_dimension = coeffs[0] # The slope is the fractal dimension D_f
             print(f"\nEstimated Fractal Dimension (Box Counting) D_f ≈ {fractal_dimension:.4f}")
        else:
             print("\nNot enough points in selected range for robust fractal dimension fit.")
             # Fallback to fitting all points if range selection failed
             if len(log_inv_epsilon) >= 2:
                 coeffs = np.polyfit(log_inv_epsilon, log_N, 1)
                 fractal_dimension = coeffs[0]
                 print(f"(Fallback fit using all points) D_f ≈ {fractal_dimension:.4f}")
             else:
                  print("\nNot enough valid data points overall for fractal dimension calculation.")

    else:
        print("\nNot enough valid data points for fractal dimension calculation.")

    # Calculate theoretical dimension
    theoretical_dim = np.log(3) / np.log(2) # N=3 copies, r=2 scaling factor
    print(f"Theoretical Fractal Dimension D_f = log(3)/log(2) ≈ {theoretical_dim:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Plot 1: Sierpinski Triangle Visualization
    plt.subplot(1, 2, 1)
    # Use smaller markers and potentially lower alpha for better visualization
    plt.scatter(sierpinski_points[:, 0], sierpinski_points[:, 1], s=0.1, c='blue', alpha=0.6)
    plt.title(f"Sierpinski Triangle ({num_points} points)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Log-Log Plot for Fractal Dimension
    plt.subplot(1, 2, 2)
    if len(box_sizes_valid) > 1:
        plt.plot(log_inv_epsilon, log_N, 'gs', markersize=4, label='Data points')
        # Plot the fitted line based on the selected range (or all points if fallback)
        if len(coeffs) > 0:
            fit_line_x = np.array([min(log_inv_epsilon), max(log_inv_epsilon)]) # Extend line across plot
            plt.plot(fit_line_x, np.polyval(coeffs, fit_line_x), 'r--',
                     label=f'Fit (Slope): $D_f \\approx {fractal_dimension:.4f}$')

        # Optional: Plot theoretical slope line for comparison (adjust intercept to match data roughly)
        C_theory = np.mean(log_N_fit) - theoretical_dim * np.mean(log_inv_epsilon_fit) if len(log_N_fit)>0 else np.mean(log_N) - theoretical_dim*np.mean(log_inv_epsilon)
        plt.plot(log_inv_epsilon, theoretical_dim * log_inv_epsilon + C_theory, 'k:', alpha=0.7,
                 label=f'Theory Slope: $D_f \\approx {theoretical_dim:.4f}$')

        plt.xlabel("$\\log_{10}(1 / \\epsilon)$")
        plt.ylabel("$\\log_{10}(N(\\epsilon))$")
        plt.title("Box Counting Log-Log Plot")
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "Not enough data for plot", ha='center', va='center')
        plt.title("Box Counting Log-Log Plot")

    plt.tight_layout()
    plt.show()