import numpy as np
import matplotlib.pyplot as plt
import time

# --- Mandelbrot Set Generation (Optimized) ---

def generate_mandelbrot(width, height, x_min, x_max, y_min, y_max, max_iter):
    """
    Generates Mandelbrot set escape time data using numpy for speed.
    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        x_min, x_max (float): Real axis bounds.
        y_min, y_max (float): Imaginary axis bounds.
        max_iter (int): Maximum number of iterations to check for escape.
    Returns:
        escape_times (np.array): A height x width array where each element
                                 is the number of iterations before escape,
                                 or max_iter if it didn't escape.
    """
    # Create coordinate grids
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    C = x[:, np.newaxis] + 1j * y[np.newaxis, :] # Complex grid C = real + i*imag

    # Initialize Z and escape times array
    Z = np.zeros_like(C, dtype=np.complex128)
    escape_times = np.full(C.shape, max_iter, dtype=np.int32)
    # Boolean mask for points still iterating
    m = np.full(C.shape, True, dtype=bool)

    print(f"Generating Mandelbrot set ({width}x{height}, max_iter={max_iter})...")
    start_time = time.time()

    for i in range(max_iter):
        # Update Z for points still iterating
        Z[m] = Z[m]**2 + C[m]
        # Find points that escaped (|Z| > 2)
        escaped = np.abs(Z) > 2
        # Update escape times for newly escaped points
        escape_times[m & escaped] = i
        # Update the mask to exclude escaped points
        m &= ~escaped
        # Optimization: if all points escaped, stop early
        if not m.any():
            break

    end_time = time.time()
    print(f"Mandelbrot generation finished in {end_time - start_time:.2f} seconds.")
    # Return transposed array so indexing is (row, col) or (y, x)
    return escape_times.T # Transpose for (height, width) indexing

# --- Boundary Identification ---

def find_boundary_pixels(escape_times, max_iter):
    """
    Identifies boundary pixels of the Mandelbrot set.
    A pixel is on the boundary if it belongs to the set (escape_time == max_iter)
    but has at least one neighbor that does not belong (escape_time < max_iter).
    Args:
        escape_times (np.array): The escape time array (height x width).
        max_iter (int): The maximum iteration count used.
    Returns:
        boundary_points (np.array): Nx2 array of [row, col] coordinates of boundary pixels.
    """
    print("Identifying boundary pixels...")
    start_time = time.time()
    height, width = escape_times.shape
    boundary_mask = np.zeros_like(escape_times, dtype=bool)

    # Check points that are IN the set (max_iter)
    set_mask = (escape_times == max_iter)

    # Check neighbors (up, down, left, right)
    # Use padding to handle edges easily
    padded_escaped = np.pad(escape_times < max_iter, pad_width=1, mode='constant', constant_values=False)

    # Check if any neighbor escaped (is True in padded_escaped)
    neighbor_escaped = (
        padded_escaped[1:-1, :-2] |  # Left
        padded_escaped[1:-1, 2:] |   # Right
        padded_escaped[:-2, 1:-1] |  # Up
        padded_escaped[2:, 1:-1]     # Down
    )

    # Boundary points are those in the set AND have a neighbor that escaped
    boundary_mask = set_mask & neighbor_escaped

    # Get the coordinates [row, col] of the boundary pixels
    boundary_points = np.argwhere(boundary_mask) # Returns Nx2 array of [row, col]

    end_time = time.time()
    print(f"Boundary identification finished in {end_time - start_time:.2f} seconds.")
    print(f"Found {len(boundary_points)} boundary points.")
    return boundary_points


# --- Box Counting Function (same as before) ---

def box_counting(points, n_scales=20):
    """
    Performs box counting on a set of 2D points (pixel coordinates).
    Args:
        points (np.array): Nx2 array of [row, col] coordinates.
        n_scales (int): Number of different box sizes (epsilon) to test.
    Returns:
        box_sizes (np.array): Array of box sizes (epsilon, in pixel units).
        box_counts (np.array): Array of corresponding box counts N(epsilon).
    """
    if points.shape[0] < 2:
        print("Warning: Not enough points for box counting.")
        return np.array([]), np.array([])

    # Use pixel coordinates directly. Min/Max define the bounding box.
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    range_coords = max_coords - min_coords

    valid_ranges = range_coords[range_coords > 0]
    if len(valid_ranges) == 0:
        max_dim_size = 1.0 # All points identical (shouldn't happen for boundary)
    else:
        # Use the larger dimension range in pixels
        max_dim_size = max(valid_ranges)

    # Smallest box size: 1 pixel seems reasonable minimum.
    min_box_size = 1.0
    # Largest box size: covers the extent of the boundary points
    max_box_size = max(max_dim_size, 1.0) # Ensure at least 1

    # Generate box sizes logarithmically from 1 pixel up to max range
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=n_scales)
    box_sizes = np.round(box_sizes).astype(int) # Use integer box sizes (pixels)
    box_sizes = np.unique(box_sizes)
    box_sizes = box_sizes[box_sizes > 0]

    if len(box_sizes) == 0:
        print("Warning: Could not generate valid positive box sizes.")
        return np.array([]), np.array([])

    box_counts = []
    actual_box_sizes = []

    print(f"Starting box counting with {len(box_sizes)} scales (pixel units)...")
    start_bc_time = time.time()

    # Pre-calculate relative coordinates
    relative_points = points - min_coords

    for epsilon in box_sizes:
        # Indices based on pixel coordinates and box size (in pixels)
        indices = np.floor(relative_points / epsilon).astype(int)
        occupied_boxes = set(map(tuple, indices))
        count = len(occupied_boxes)

        if count > 0:
            box_counts.append(count)
            actual_box_sizes.append(epsilon)

    end_bc_time = time.time()
    print(f"Box counting finished in {end_bc_time - start_bc_time:.2f} seconds.")

    return np.array(actual_box_sizes), np.array(box_counts)

# --- Main Simulation and Plotting ---

if __name__ == "__main__":
    # Parameters
    img_width = 800
    img_height = 800
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iterations = 150 # Increase for finer detail, but slower
    num_scales_bc = 20  # Number of box sizes for counting

    # 1. Generate Mandelbrot Data
    escape_data = generate_mandelbrot(img_width, img_height, x_min, x_max, y_min, y_max, max_iterations)

    # 2. Identify Boundary Pixels
    boundary_pixel_coords = find_boundary_pixels(escape_data, max_iterations)

    # Check if boundary points were found
    if boundary_pixel_coords.shape[0] < 2:
        print("Error: Could not find enough boundary points. Try increasing resolution or max_iterations.")
    else:
        # 3. Perform Box Counting on Boundary Pixel Coordinates
        box_sizes, box_counts = box_counting(boundary_pixel_coords, n_scales=num_scales_bc)

        # 4. Log-Log Analysis and Fitting
        valid_indices = (box_counts > 1) & (box_sizes > 0)
        box_sizes_valid = box_sizes[valid_indices]
        box_counts_valid = box_counts[valid_indices]

        fractal_dimension = np.nan
        coeffs = []
        log_inv_epsilon = np.array([])
        log_N = np.array([])

        if len(box_sizes_valid) > 1:
            log_inv_epsilon = np.log10(1.0 / box_sizes_valid) # Epsilon is in pixels here
            log_N = np.log10(box_counts_valid)

            # Fit a line - try excluding smallest/largest scales
            fit_start_index = int(len(log_inv_epsilon) * 0.1) # Skip smallest boxes (pixel effects)
            fit_end_index = int(len(log_inv_epsilon) * 0.9)   # Skip largest boxes (finite size effects)
            fit_start_index = max(0, fit_start_index)
            fit_end_index = min(len(log_inv_epsilon), fit_end_index)

            if fit_end_index - fit_start_index >= 2:
                 log_inv_epsilon_fit = log_inv_epsilon[fit_start_index:fit_end_index]
                 log_N_fit = log_N[fit_start_index:fit_end_index]
                 coeffs = np.polyfit(log_inv_epsilon_fit, log_N_fit, 1)
                 fractal_dimension = coeffs[0]
                 print(f"\nEstimated Fractal Dimension (Box Counting on Boundary) D_f ≈ {fractal_dimension:.4f}")
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

        # Theoretical Dimension Note
        print("Theoretical Hausdorff Dimension of the Mandelbrot Set Boundary = 2.0")

        # --- Plotting ---
        plt.figure(figsize=(14, 7))

        # Plot 1: Mandelbrot Boundary Visualization
        ax1 = plt.subplot(1, 2, 1)
        # Plot only the boundary points found
        # Need to flip y-axis because array rows increase downwards, but plot y increases upwards
        ax1.scatter(boundary_pixel_coords[:, 1], img_height - 1 - boundary_pixel_coords[:, 0], s=0.1, c='black')
        ax1.set_title(f"Mandelbrot Set Boundary ({len(boundary_pixel_coords)} points)")
        ax1.set_xlabel("Pixel Column (x)")
        ax1.set_ylabel("Pixel Row (y - flipped)")
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlim(0, img_width)
        ax1.set_ylim(0, img_height)
        ax1.grid(True, linestyle='--', alpha=0.5)


        # Plot 2: Log-Log Plot for Fractal Dimension
        ax2 = plt.subplot(1, 2, 2)
        if len(log_inv_epsilon) > 0 and len(log_N) > 0:
            ax2.plot(log_inv_epsilon, log_N, 'bo', markersize=4, label='Data points (Boundary Pixels)')

            if len(coeffs) > 0:
                # Plot fitted line over the range it was fitted on
                if fit_end_index - fit_start_index >= 2:
                     fit_line_x = log_inv_epsilon_fit
                else:
                     fit_line_x = log_inv_epsilon # Fallback if fit used all points

                sorted_indices = np.argsort(fit_line_x)
                fit_line_x_sorted = fit_line_x[sorted_indices]
                fit_line_y = np.polyval(coeffs, fit_line_x_sorted)

                ax2.plot(fit_line_x_sorted, fit_line_y, 'r--',
                         label=f'Fit (Slope): $D_f \\approx {fractal_dimension:.4f}$')

                # Highlight points used for fit
                if fit_end_index - fit_start_index >= 2:
                     ax2.plot(log_inv_epsilon_fit, log_N_fit, 'rs', markersize=5, fillstyle='none',
                              label='Points used for fit')

            ax2.set_xlabel("$\\log_{10}(1 / \\epsilon)$  ($\\epsilon$ in pixels)")
            ax2.set_ylabel("$\\log_{10}(N(\\epsilon))$")
            ax2.set_title("Box Counting Log-Log Plot (Boundary)")
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "Not enough data for plot", ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Box Counting Log-Log Plot (Boundary)")

        plt.tight_layout()
        plt.show()