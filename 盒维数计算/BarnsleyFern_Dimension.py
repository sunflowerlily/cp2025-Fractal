import numpy as np
import matplotlib.pyplot as plt
import random
import time # To measure time

# --- Barnsley's Fern Generation (IFS) ---

def barnsley_fern_ifs(n_points):
    """
    Generates points for Barnsley's Fern using its Iterated Function System (IFS).
    Args:
        n_points (int): The number of points to generate (excluding initial burn-in).
    Returns:
        np.array: An (n_points)x2 numpy array of points approximating Barnsley's Fern.
    """
    # IFS transformation parameters [a, b, c, d, e, f] and probability p
    # From Wikipedia: https://en.wikipedia.org/wiki/Barnsley_fern
    ifs_params = [
        # f1: Stem
        {'coeffs': [0.00, 0.00, 0.00, 0.16, 0.00, 0.00], 'p': 0.01},
        # f2: Successively smaller leaflets
        {'coeffs': [0.85, 0.04, -0.04, 0.85, 0.00, 1.60], 'p': 0.85},
        # f3: Largest left-hand leaflet
        {'coeffs': [0.20, -0.26, 0.23, 0.22, 0.00, 1.60], 'p': 0.07},
        # f4: Largest right-hand leaflet
        {'coeffs': [-0.15, 0.28, 0.26, 0.24, 0.00, 0.44], 'p': 0.07}
    ]

    # Calculate cumulative probabilities for selection
    cumulative_p = np.cumsum([f['p'] for f in ifs_params])

    points = np.zeros((n_points, 2))
    # Start at the origin
    x, y = 0.0, 0.0

    # Burn-in period (let points settle onto the attractor)
    burn_in = 100
    for _ in range(burn_in):
        rand_val = random.random() # Random number between 0.0 and 1.0
        # Select transformation based on probability
        for i, p_cum in enumerate(cumulative_p):
            if rand_val <= p_cum:
                params = ifs_params[i]['coeffs']
                a, b, c, d, e, f = params
                # Apply transformation
                x_new = a * x + b * y + e
                y_new = c * x + d * y + f
                x, y = x_new, y_new
                break # Exit inner loop once transformation is applied

    # Generate the points for the dataset
    for i in range(n_points):
        rand_val = random.random()
        for j, p_cum in enumerate(cumulative_p):
             if rand_val <= p_cum:
                params = ifs_params[j]['coeffs']
                a, b, c, d, e, f = params
                x_new = a * x + b * y + e
                y_new = c * x + d * y + f
                x, y = x_new, y_new
                points[i] = [x, y] # Store the new point
                break

    return points

# --- Box Counting Function (same as before) ---

def box_counting(points, n_scales=20):
    """
    Performs box counting on a set of 2D points.
    Args:
        points (np.array): Nx2 array of [x, y] coordinates.
        n_scales (int): Number of different box sizes (epsilon) to test,
                        distributed logarithmically.
    Returns:
        box_sizes (np.array): Array of box sizes (epsilon) used.
        box_counts (np.array): Array of corresponding box counts N(epsilon).
    """
    if points.shape[0] < 2:
        return np.array([]), np.array([])

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    range_coords = max_coords - min_coords

    # Handle cases where range might be zero in one dimension
    valid_ranges = range_coords[range_coords > 0]
    if len(valid_ranges) == 0: # All points identical
        max_dim_size = 1.0
    else:
        max_dim_size = max(valid_ranges)

    # Define scale range
    # Smallest box size should ideally be related to point density or minimum separation
    # Using a power-of-2 division of the max range is common
    min_box_size = max_dim_size / (2**n_scales)
    if min_box_size <= np.finfo(float).eps: # Avoid zero or negative box size
        min_box_size = np.finfo(float).eps * 10

    max_box_size = max_dim_size

    # Generate box sizes logarithmically
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=n_scales)
    box_sizes = np.unique(box_sizes) # Ensure unique sizes
    box_sizes = box_sizes[box_sizes > 0] # Ensure positive sizes

    if len(box_sizes) == 0:
        print("Warning: Could not generate valid positive box sizes.")
        return np.array([]), np.array([])

    box_counts = []
    actual_box_sizes = []

    start_bc_time = time.time()
    print(f"Starting box counting with {len(box_sizes)} scales...")

    # Pre-calculate relative coordinates to avoid repeated subtraction
    relative_points = points - min_coords

    for i, epsilon in enumerate(box_sizes):
        # Use floor division to get box indices. Add small value to handle boundary? No, floor is fine.
        # indices = np.floor((points - min_coords) / epsilon).astype(int) # Original way
        indices = np.floor(relative_points / epsilon).astype(int) # Slightly faster

        # Use a set to store unique box indices (represented as tuples)
        occupied_boxes = set(map(tuple, indices))
        count = len(occupied_boxes)

        if count > 0:
            box_counts.append(count)
            actual_box_sizes.append(epsilon)
        # Optional: Print progress
        # if (i + 1) % 5 == 0:
        #     print(f"  Scale {i+1}/{len(box_sizes)}, Epsilon: {epsilon:.4e}, Count: {count}")


    end_bc_time = time.time()
    print(f"Box counting finished in {end_bc_time - start_bc_time:.2f} seconds.")

    return np.array(actual_box_sizes), np.array(box_counts)

# --- Main Simulation and Plotting ---

if __name__ == "__main__":
    # Parameters
    num_points = 100000  # Number of points for the IFS generation (more points = better detail)
    num_scales_bc = 10 # Number of scales for box counting

    # 1. Generate Barnsley's Fern Points
    print(f"Generating {num_points} Barnsley's Fern points using IFS...")
    start_gen_time = time.time()
    fern_points = barnsley_fern_ifs(num_points)
    end_gen_time = time.time()
    print(f"Point generation complete in {end_gen_time - start_gen_time:.2f} seconds.")

    # 2. Perform Box Counting
    box_sizes, box_counts = box_counting(fern_points, n_scales=num_scales_bc)

    # Filter out potential issues for log-log plot (need at least 2 points for fit)
    valid_indices = (box_counts > 1) & (box_sizes > 0)
    box_sizes_valid = box_sizes[valid_indices]
    box_counts_valid = box_counts[valid_indices]

    fractal_dimension = np.nan # Default value
    coeffs = []
    log_inv_epsilon = np.array([])
    log_N = np.array([])

    if len(box_sizes_valid) > 1:
        # Calculate log values: log(N) vs log(1/epsilon) (using base 10)
        log_inv_epsilon = np.log10(1.0 / box_sizes_valid)
        log_N = np.log10(box_counts_valid)

        # Fit a line (degree 1 polynomial) to the log-log data
        # Select a good range for fitting (avoid smallest/largest scales if noisy)
        # Heuristic: use middle ~70% of points (adjust as needed)
        fit_start_index = int(len(log_inv_epsilon) * 0.15)
        fit_end_index = int(len(log_inv_epsilon) * 0.85)
        # Ensure range indices are valid and provide at least 2 points
        fit_start_index = max(0, fit_start_index)
        fit_end_index = min(len(log_inv_epsilon), fit_end_index)

        if fit_end_index - fit_start_index >= 2:
             log_inv_epsilon_fit = log_inv_epsilon[fit_start_index:fit_end_index]
             log_N_fit = log_N[fit_start_index:fit_end_index]
             coeffs = np.polyfit(log_inv_epsilon_fit, log_N_fit, 1)
             fractal_dimension = coeffs[0] # The slope is the fractal dimension D_f
             print(f"\nEstimated Fractal Dimension (Box Counting, fitted range) D_f ≈ {fractal_dimension:.4f}")
        else:
             print("\nWarning: Not enough points in selected range for robust fit. Fitting all points.")
             if len(log_inv_epsilon) >= 2:
                 coeffs = np.polyfit(log_inv_epsilon, log_N, 1)
                 fractal_dimension = coeffs[0]
                 print(f"(Fit using all points) D_f ≈ {fractal_dimension:.4f}")
             else:
                  print("\nError: Not enough valid data points overall for fractal dimension calculation.")

    else:
        print("\nError: Not enough valid data points (box_counts > 1) for fractal dimension calculation.")

    # Theoretical Dimension Note:
    # Unlike simple self-similar fractals (Koch, Sierpinski), Barnsley's Fern involves
    # different scaling factors and transformations (rotations, shears).
    # There isn't a straightforward theoretical formula like log(N)/log(r).
    # Numerical estimates often place its dimension around 1.7-1.8, but it's primarily
    # determined numerically, like through box counting.
    print("Note: A simple theoretical formula for Barnsley Fern's dimension is not readily available due to its complex IFS.")
    print("      Numerical estimates are typically around 1.7 - 1.8.")


    # --- Plotting ---
    plt.figure(figsize=(12, 8)) # Adjusted figure size

    # Plot 1: Barnsley's Fern Visualization
    ax1 = plt.subplot(1, 2, 1)
    # Use smaller markers and green color for fern look
    ax1.scatter(fern_points[:, 0], fern_points[:, 1], s=0.05, c='green', alpha=0.7)
    ax1.set_title(f"Barnsley's Fern ({num_points} points)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect('equal', adjustable='box') # Maintain aspect ratio
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Log-Log Plot for Fractal Dimension
    ax2 = plt.subplot(1, 2, 2)
    if len(log_inv_epsilon) > 0 and len(log_N) > 0:
        ax2.plot(log_inv_epsilon, log_N, 'bo', markersize=4, label='Data points')

        # Plot the fitted line based on the selected range (or all points if fallback)
        if len(coeffs) > 0:
            # Generate line points covering the range used for fitting or the whole range
            if fit_end_index - fit_start_index >= 2:
                 fit_line_x = log_inv_epsilon_fit
            else:
                 fit_line_x = log_inv_epsilon

            # Ensure fit_line_x is sorted for plotting the line correctly
            sorted_indices = np.argsort(fit_line_x)
            fit_line_x_sorted = fit_line_x[sorted_indices]
            fit_line_y = np.polyval(coeffs, fit_line_x_sorted)

            ax2.plot(fit_line_x_sorted, fit_line_y, 'r--',
                     label=f'Fit (Slope): $D_f \\approx {fractal_dimension:.4f}$')

            # Highlight the points used for fitting
            if fit_end_index - fit_start_index >= 2:
                 ax2.plot(log_inv_epsilon_fit, log_N_fit, 'rs', markersize=5, fillstyle='none',
                          label='Points used for fit')


        ax2.set_xlabel("$\\log_{10}(1 / \\epsilon)$")
        ax2.set_ylabel("$\\log_{10}(N(\\epsilon))$")
        ax2.set_title("Box Counting Log-Log Plot")
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, "Not enough data for plot", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Box Counting Log-Log Plot")

    plt.tight_layout()
    plt.show()