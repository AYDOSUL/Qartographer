import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy.optimize import minimize
import json
import os

# Radius of the circles representing qubits in the lattice.
radius = 1e-5

# Define the drive points in the lattice. These are now fixed.
fixed_drive_points = np.array([
[
    0.00003,
    -0.00002
],
[
    0.00013,
    -0.00002
],
[
    0.00023,
    -0.00002
],
[
    0.00033,
    -0.00002
],
[
    0.00043,
    -0.00002
],
[
    0.00053,
    -0.00002
],
[
    0.00063,
    -0.00002
],
[
    0.00073,
    -0.00002
],
[
    0.00083,
    -0.00002
],
[
    0.00093,
    -0.00002
],
[
    0.00103,
    -0.00002
],
[
    0.00113,
    -0.00002
],
[
    0.00123,
    -0.00002
],
[
    0.00133,
    -0.00002
],
[
    0.00143,
    -0.00002
],
[
    0.00153,
    -0.00002
],
[
    0.00153,
    0.00008
],
[
    0.00153,
    0.00018
],
[
    0.00153,
    0.00028
],
[
    0.00153,
    0.00038
],
[
    0.00153,
    0.00088
],
[
    0.00153,
    0.00078
],
[
    0.00153,
    0.00068
],
[
    0.00153,
    0.00058
],
[
    0.00153,
    0.00048
]
])

# Radius of the circles representing drive points in the lattice.
drive_radius = 3e-5
# Size for multiplexer points (larger for visibility)
multiplexer_size = 1e-4

# Ideal length of the readout lines (from qubit to multiplexer), can be adjusted.
Ideal_readout_line_length = 1e-4

# Define initial multiplexer lines (start and end points)
num_multiplexers = 3
initial_multiplexer_lines = np.array([
    [[0.0001, 0.00005], [0.0001, 0.0002]],
    [[0.0001, 0.00006], [0.0001, 0.0002]],
    [[0.0001, 0.00007], [0.0001, 0.0002]],
])

# Resolution for drive lines (number of intermediate points)
resolution = 1

# Optimization constants
ALPHA_PROXIMITY_MULTI_DRIVE = 0 # Penalty for multiplexer endpoints near drive points
GAMMA_LINE_LENGTH = 50000 # Penalty for the total length of a multiplexer line
DELTA_WEIGHT_READOUT_LINE = 10000000

# Constants for the drive line cost function
PENALTY_DRIVE_LENGTH = 10.0
PENALTY_DRIVE_QUBIT_PROXIMITY = 10.0
PENALTY_DRIVE_MULTI_PROXIMITY = 1.0
PENALTY_DRIVE_DRIVE_PROXIMITY = 1000.0
MIN_PROXIMITY_DISTANCE = 1e-6 # Minimum allowed distance to other components

# A small epsilon value to prevent division from zero in proximity calculations.
EPSILON_DISTANCE = 1e-6

def load_qubit_data_from_json():
    """
    Prompts the user for a JSON filename and loads qubit coordinates and couplings.
    """
    while True:
        filename = input("Enter the JSON filename containing qubit data (e.g., optimized_qubit_layout.json): ")
        if not filename.strip():
            print("Filename cannot be empty. Please try again.")
            continue
        if not filename.endswith(".json"):
            filename += ".json"
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Extract qubit coordinates
            loaded_qubit_coords_dict = data.get("optimized_qubit_coordinates", {})
            loaded_qubit_coords = []
            # Sort qubits by their index (Qubit_0, Qubit_1, etc.)
            sorted_qubit_keys = sorted(loaded_qubit_coords_dict.keys(), key=lambda x: int(x.split('_')[1]))
            for key in sorted_qubit_keys:
                coord = loaded_qubit_coords_dict[key]
                loaded_qubit_coords.append([coord['x'], coord['y']])
            
            # Extract couplings
            loaded_qubit_couplings_list = data.get("optimized_couplings", [])
            loaded_qubit_couplings = []
            for coupling in loaded_qubit_couplings_list:
                loaded_qubit_couplings.append([coupling['qubit1_index'], coupling['qubit2_index']])

            print(f"\nSuccessfully loaded data from '{filename}'.")
            return np.array(loaded_qubit_coords), np.array(loaded_qubit_couplings)

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found. Please check the filename and try again.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{filename}'. Please ensure it's a valid JSON file.")
        except KeyError as e:
            print(f"Error: Missing expected key in JSON file: {e}. Ensure 'optimized_qubit_coordinates' and 'optimized_couplings' are present.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def point_to_line_segment_distance(p, a, b):
    p, a, b = np.array(p), np.array(a), np.array(b)
    ab = b - a
    ap = p - a
    
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq == 0:
        return np.linalg.norm(p - a), a
    
    t = np.dot(ap, ab) / ab_len_sq
    
    if t <= 0.0:
        closest_point = a
    elif t >= 1.0:
        closest_point = b
    else:
        closest_point = a + t * ab
    
    distance = np.linalg.norm(p - closest_point)
    return distance, closest_point


def plot_lattice(ax_object, drive_points, drive_paths, multiplexer_lines, read_to_multi_lines, qubits, couplings, title="Lattice Plot"):
    ax_object.clear()

    # Plotting Qubits
    for index in range(len(qubits)):
        qubit_data = qubits[index]
        circle = Circle((qubit_data[0], qubit_data[1]), radius, color='blue', fill=True, label='Qubit')
        ax_object.add_patch(circle)

    # Plotting Drive Points (Yellow Circles)
    for point_data in drive_points:
        circle = Circle((point_data[0], point_data[1]), drive_radius, color='yellow', fill=True, label='Control/Readout Ports')
        ax_object.add_patch(circle)

    # Plotting Qubit Couplings
    for coupling in couplings:
        start_qubit_index = coupling[0]
        end_qubit_index = coupling[1]

        if start_qubit_index < len(qubits) and end_qubit_index < len(qubits):
            start_coords = qubits[start_qubit_index]
            end_coords = qubits[end_qubit_index]
            ax_object.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], color='red', label='Qubit Coupler')
        else:
            print(f"Warning: Coupling {coupling} refers to out-of-bounds qubit index.")

    # Plotting Drive Lines (Green Lines)
    for path in drive_paths:
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        ax_object.plot(x_coords, y_coords, color='green', linewidth=2, label='RF Control Line')

    # Plotting Multiplexer Lines (Solid Black Lines)
    for line_info in multiplexer_lines:
        start_coords = line_info[0]
        end_coords = line_info[1]
        ax_object.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], color='black', linewidth=2, label='Readout Multiplexer')

    # Plotting Readout Resonator Lines (Orange Lines)
    for line_info in read_to_multi_lines:
        qubit_index = line_info[0]
        closest_point_on_line = line_info[1]
        
        start_coords = qubits[qubit_index]
        end_coords = closest_point_on_line
        ax_object.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], color='orange', linewidth=2, label='Readout Line')

    ax_object.set_aspect('equal', adjustable='box')
    ax_object.grid(True)
    ax_object.set_xlabel("X-coordinate")
    ax_object.set_ylabel("Y-coordinate")
    ax_object.set_title(title)
    
    handles, labels = ax_object.get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(l)] for l in unique_labels]
    ax_object.legend(unique_handles, unique_labels, loc='best')

def create_initial_drive_paths(qubit_coords, fixed_drive_points, resolution):
    drive_paths = []
    initial_intermediate_points = []
    for i in range(len(qubit_coords)):
        qubit_coord = qubit_coords[i]
        drive_point_coord = fixed_drive_points[i]
        path = [qubit_coord]
        
        for j in range(1, resolution + 1):
            t = j / (resolution + 1)
            intermediate_point = qubit_coord + t * (drive_point_coord - qubit_coord)
            path.append(intermediate_point)
            initial_intermediate_points.append(intermediate_point)

        path.append(drive_point_coord)
        drive_paths.append(path)
    
    return np.array(drive_paths, dtype=object), np.array(initial_intermediate_points)

# Drive cost function with proximity and length penalties
def drive_cost_function(intermediate_points_flat, qubit_coords_in_cost_fn, fixed_drive_points_in_cost_fn, resolution_in_cost_fn, fixed_multiplexer_lines_for_penalty):
    num_qubits = len(qubit_coords_in_cost_fn)
    intermediate_points_reshaped = intermediate_points_flat.reshape(num_qubits, resolution_in_cost_fn, 2)
    total_cost = 0.0
    all_drive_paths = []

    # Calculate penalties for each drive line
    for i in range(num_qubits):
        qubit_coord = qubit_coords_in_cost_fn[i]
        drive_point_coord = fixed_drive_points_in_cost_fn[i]
        path_points = [qubit_coord] + list(intermediate_points_reshaped[i]) + [drive_point_coord]
        all_drive_paths.append(path_points)

        # Total length penalty
        drive_line_length = 0.0
        for j in range(len(path_points) - 1):
            drive_line_length += np.linalg.norm(path_points[j] - path_points[j+1])
        total_cost += PENALTY_DRIVE_LENGTH * drive_line_length

        # Proximity penalty to other qubits (not the connected one)
        for other_qubit_index in range(num_qubits):
            if other_qubit_index != i:
                other_qubit_coord = qubit_coords_in_cost_fn[other_qubit_index]
                for p_index in range(1, len(path_points) - 1): # exclude start and end points
                    point_on_drive_line = path_points[p_index]
                    dist_to_qubit = np.linalg.norm(point_on_drive_line - other_qubit_coord)
                    if dist_to_qubit < MIN_PROXIMITY_DISTANCE:
                        total_cost += PENALTY_DRIVE_QUBIT_PROXIMITY / (dist_to_qubit + EPSILON_DISTANCE)**2

        # Proximity penalty to multiplexer lines
        for multi_line in fixed_multiplexer_lines_for_penalty:
            start_multi, end_multi = multi_line
            for p_index in range(1, len(path_points)):
                p_current = path_points[p_index]
                p_previous = path_points[p_index - 1]
                
                # Simplified check for proximity to the multi-line segment
                dist_to_multi_segment, _ = point_to_line_segment_distance(p_current, start_multi, end_multi)
                if dist_to_multi_segment < MIN_PROXIMITY_DISTANCE:
                    total_cost += PENALTY_DRIVE_MULTI_PROXIMITY / (dist_to_multi_segment + EPSILON_DISTANCE)**2

    # Proximity penalty between drive lines
    for i in range(num_qubits):
        for k in range(i + 1, num_qubits):
            # Compare each point on drive line i with each point on drive line k
            for p_i in all_drive_paths[i][1:-1]: # exclude endpoints
                for p_k in all_drive_paths[k][1:-1]:
                    dist_between_paths = np.linalg.norm(p_i - p_k)
                    if dist_between_paths < MIN_PROXIMITY_DISTANCE:
                        total_cost += PENALTY_DRIVE_DRIVE_PROXIMITY / (dist_between_paths + EPSILON_DISTANCE)**2

    return total_cost

def multiplexer_cost_function(params_flat, qubit_coords_for_cost, fixed_drive_points_for_penalty, num_qubits, num_multiplexers):
    multiplexer_lines = params_flat.reshape(num_multiplexers, 2, 2)
    total_cost = 0.0

    # Cost for readout line connections (qubit to closest point on closest line)
    for i in range(num_qubits):
        qubit_coord = qubit_coords_for_cost[i]
        min_dist_to_multi = float('inf')

        for j in range(num_multiplexers):
            start_point = multiplexer_lines[j, 0]
            end_point = multiplexer_lines[j, 1]
            dist, _ = point_to_line_segment_distance(qubit_coord, start_point, end_point)
            if dist < min_dist_to_multi:
                min_dist_to_multi = dist
        
        total_cost += DELTA_WEIGHT_READOUT_LINE * ((min_dist_to_multi - Ideal_readout_line_length)**2)

    # Penalty for the total length of each multiplexer line
    for j in range(num_multiplexers):
        start_point = multiplexer_lines[j, 0]
        end_point = multiplexer_lines[j, 1]
        line_length = np.linalg.norm(start_point - end_point)
        total_cost += GAMMA_LINE_LENGTH * (line_length**2)

    # Proximity penalty (multiplexer line endpoints to drive points)
    proximity_penalty_multi_drive = 0.0
    all_multi_endpoints = multiplexer_lines.reshape(-1, 2)
    for i in range(len(all_multi_endpoints)):
        for j in range(len(fixed_drive_points_for_penalty)):
            multi_endpoint = all_multi_endpoints[i]
            drive_point = fixed_drive_points_for_penalty[j]
            distance = np.linalg.norm(multi_endpoint - drive_point)
            proximity_penalty_multi_drive += ALPHA_PROXIMITY_MULTI_DRIVE / ((distance + EPSILON_DISTANCE)**2)
    total_cost += proximity_penalty_multi_drive

    return total_cost

def optimize_drive_paths(initial_intermediate_points, qubit_coords_for_optimization, fixed_drive_points_for_optimization, resolution_for_optimization, optimized_multiplexer_lines_for_penalty):
    initial_intermediate_points_flat = initial_intermediate_points.flatten()
    result = minimize(drive_cost_function, initial_intermediate_points_flat, args=(qubit_coords_for_optimization, fixed_drive_points_for_optimization, resolution_for_optimization, optimized_multiplexer_lines_for_penalty), method='L-BFGS-B')
    return result.x.reshape(-1, 2)

def optimize_multiplexer_points(initial_multiplexer_lines, qubit_coords_for_optimization, current_drive_points):
    num_qubits = len(qubit_coords_for_optimization)
    num_multiplexers = len(initial_multiplexer_lines)
    initial_params_flat = initial_multiplexer_lines.flatten()
    result = minimize(multiplexer_cost_function, initial_params_flat,
                      args=(qubit_coords_for_optimization, current_drive_points, num_qubits, num_multiplexers),
                      method='L-BFGS-B')

    optimized_multiplexer_lines = result.x.reshape(num_multiplexers, 2, 2)

    read_to_multi_lines = []
    for i in range(num_qubits):
        qubit_coord = qubit_coords_for_optimization[i]
        min_dist_to_multi = float('inf')
        closest_point_on_line = None
        for j in range(num_multiplexers):
            start_point = optimized_multiplexer_lines[j, 0]
            end_point = optimized_multiplexer_lines[j, 1]
            dist, current_closest_point = point_to_line_segment_distance(qubit_coord, start_point, end_point)
            if dist < min_dist_to_multi:
                min_dist_to_multi = dist
                closest_point_on_line = current_closest_point
        read_to_multi_lines.append([i, closest_point_on_line])

    return optimized_multiplexer_lines, np.array(read_to_multi_lines, dtype=object)

def save_optimized_layout_to_json(drive_points, multiplexer_lines, read_to_multi_lines, drive_paths):
    while True:
        filename = input("Enter the desired filename for the optimized layout data (e.g., optimized_layout_details.json): ")
        if not filename.strip():
            print("Filename cannot be empty. Please try again.")
            continue
        if not filename.endswith(".json"):
            filename += ".json"
        break

    # Convert NumPy arrays to lists for JSON serialization
    formatted_drive_points = drive_points.tolist()
    formatted_multiplexer_lines = multiplexer_lines.tolist()
    
    # read_to_multi_lines contains qubit index and a coordinate.
    formatted_read_to_multi_lines = []
    for line_info in read_to_multi_lines:
        formatted_read_to_multi_lines.append([int(line_info[0]), line_info[1].tolist()])

    # drive_paths is a list of lists of arrays, need to convert inner arrays to lists
    formatted_drive_paths = []
    for path in drive_paths:
        formatted_path = [point.tolist() for point in path]
        formatted_drive_paths.append(formatted_path)

    data_to_save = {
        "fixed_drive_points": formatted_drive_points,
        "optimized_multiplexer_lines": formatted_multiplexer_lines,
        "optimized_readout_to_multiplexer_lines": formatted_read_to_multi_lines,
        "optimized_drive_paths": formatted_drive_paths
    }

    try:
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"\nSuccessfully saved optimized layout data to '{filename}'.")
        print("-" * 50)
        print(json.dumps(data_to_save, indent=4))
        print("-" * 50)
    except IOError as e:
        print(f"Error: Could not write to file '{filename}'. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving JSON: {e}")

# Load qubit data from JSON file
qubit_coords, qubit_couplings = load_qubit_data_from_json()

# Ensure the number of fixed drive points matches the number of qubits loaded
if len(fixed_drive_points) != len(qubit_coords):
    print(f"Warning: Number of fixed_drive_points ({len(fixed_drive_points)}) does not match number of qubits loaded ({len(qubit_coords)}). This might lead to unexpected behavior.")
    if len(fixed_drive_points) > len(qubit_coords):
        fixed_drive_points = fixed_drive_points[:len(qubit_coords)]
        print(f"Truncated fixed_drive_points to {len(fixed_drive_points)}.")
    else:
        print(f"Not enough fixed_drive_points for all qubits. Some qubits might not have a corresponding drive point.")


drive_points_fixed = fixed_drive_points

print("--- Starting Optimization for Multiplexer Lines ---")
optimized_multiplexer_lines, optimized_read_to_multi_lines = \
    optimize_multiplexer_points(initial_multiplexer_lines, qubit_coords, drive_points_fixed)
print("Optimized Multiplexer Lines:\n", optimized_multiplexer_lines)

print("\n--- Starting Optimization for Drive Paths ---")
initial_drive_paths, initial_intermediate_points = create_initial_drive_paths(qubit_coords, drive_points_fixed, resolution)

# Check if initial_intermediate_points is empty, which can happen if qubit_coords is empty
if initial_intermediate_points.size == 0:
    print("No intermediate points to optimize. Skipping drive path optimization.")
    optimized_intermediate_points = np.array([])
    optimized_drive_paths = []
else:
    optimized_intermediate_points = optimize_drive_paths(
        initial_intermediate_points,
        qubit_coords,
        drive_points_fixed,
        resolution,
        optimized_multiplexer_lines
    )

    # Reconstruct the optimized drive paths for plotting
    optimized_drive_paths = []
    optimized_intermediate_points_reshaped = optimized_intermediate_points.reshape(len(qubit_coords), resolution, 2)
    for i in range(len(qubit_coords)):
        path = [qubit_coords[i]] + list(optimized_intermediate_points_reshaped[i]) + [drive_points_fixed[i]]
        optimized_drive_paths.append(path)
    print("Optimized Intermediate Drive Points (reshaped):\n", optimized_intermediate_points_reshaped)

# Plot the optimized lattice
fig_optimized, ax_optimized = plt.subplots(figsize=(10, 8))
plot_lattice(ax_optimized, drive_points_fixed, optimized_drive_paths,
              optimized_multiplexer_lines, optimized_read_to_multi_lines,
              qubit_coords, qubit_couplings, title=f"Optimized Lattice (Multiplexer First)")
plt.show()

# Save the optimized layout data to a JSON file
save_optimized_layout_to_json(
    drive_points_fixed,
    optimized_multiplexer_lines,
    optimized_read_to_multi_lines,
    optimized_drive_paths
)
