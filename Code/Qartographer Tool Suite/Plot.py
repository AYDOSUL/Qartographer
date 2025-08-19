import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

# Visual constants for plotting
radius = 1e-6       # Radius of the circles representing qubits
drive_radius = 1e-2 # Radius of the circles representing drive points

def plot_merged_lattice(ax_object, data, title="Optimized Quantum Chip Layout"):
    ax_object.clear()

    # Extract data from the merged dictionary
    # Qubit coordinates and couplings (from the first optimization script)
    loaded_qubit_coords_dict = data.get("optimized_qubit_coordinates", {})
    qubits = []
    # Sort qubits by their index (Qubit_0, Qubit_1, etc.) to maintain order
    sorted_qubit_keys = sorted(loaded_qubit_coords_dict.keys(), key=lambda x: int(x.split('_')[1]))
    for key in sorted_qubit_keys:
        coord = loaded_qubit_coords_dict[key]
        qubits.append([coord['x'], coord['y']])
    qubits = np.array(qubits)

    loaded_qubit_couplings_list = data.get("optimized_couplings", [])
    couplings = []
    for coupling in loaded_qubit_couplings_list:
        couplings.append([coupling['qubit1_index'], coupling['qubit2_index']])
    couplings = np.array(couplings)

    # Control line data (from the second optimization script)
    drive_points = np.array(data.get("fixed_drive_points", []))
    multiplexer_lines = np.array(data.get("optimized_multiplexer_lines", []))
    read_to_multi_lines_raw = data.get("optimized_readout_to_multiplexer_lines", [])
    # Convert readout lines to a more usable format (qubit_index, closest_point_coords)
    read_to_multi_lines = []
    for line_info in read_to_multi_lines_raw:
        read_to_multi_lines.append([line_info[0], np.array(line_info[1])])

    drive_paths = data.get("optimized_drive_paths", []) # This is a list of lists of lists, keep as is for iteration

    # Plotting Qubits
    for index in range(len(qubits)):
        qubit_data = qubits[index]
        circle = Circle((qubit_data[0], qubit_data[1]), radius, color='blue', fill=True, label='Qubit')
        ax_object.add_patch(circle)

    # Plotting Drive Points (Yellow Circles)
    for point_data in drive_points:
        circle = Circle((point_data[0], point_data[1]), drive_radius, color='yellow', fill=True, label='Control/Readout Port')
        ax_object.add_patch(circle)

    # Plotting Qubit Couplings
    for coupling in couplings:
        start_qubit_index = coupling[0]
        end_qubit_index = coupling[1]

        if start_qubit_index < len(qubits) and end_qubit_index < len(qubits):
            start_coords = qubits[start_qubit_index]
            end_coords = qubits[end_qubit_index]
            ax_object.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]],
                           color='red', label='Qubit Couplers')
        else:
            print(f"Warning: Coupling {coupling} refers to out-of-bounds qubit index.")

    # Plotting Drive Lines (Green Lines)
    for path in drive_paths:
        # Ensure path points are numpy arrays for consistent operations
        path_np = [np.array(p) for p in path]
        x_coords = [p[0] for p in path_np]
        y_coords = [p[1] for p in path_np]
        ax_object.plot(x_coords, y_coords, color='green', linewidth=2, label='RF Control Line')

    # Plotting Multiplexer Lines (Solid Black Lines)
    for line_info in multiplexer_lines:
        start_coords = np.array(line_info[0])
        end_coords = np.array(line_info[1])
        ax_object.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]],
                       color='black', linewidth=2, label='Readout Multiplexer')

    # Plotting Readout Resonator Lines (Orange Lines)
    for line_info in read_to_multi_lines:
        qubit_index = line_info[0]
        closest_point_on_line = line_info[1] # This is already a numpy array from conversion above

        start_coords = qubits[qubit_index]
        end_coords = closest_point_on_line
        ax_object.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]],
                       color='orange', linewidth=2, label='Readout Line')

    ax_object.set_aspect('equal', adjustable='box')
    ax_object.grid(True)
    ax_object.set_xlabel("X-coordinate")
    ax_object.set_ylabel("Y-coordinate")
    ax_object.set_title(title)

    # Create a single legend with unique labels
    handles, labels = ax_object.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for i, label in enumerate(labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handles[i])
    ax_object.legend(unique_handles, unique_labels, loc='best')


def plot_merged_layout_from_file():
    print("--- Plot Device Map ---")

    while True:
        filename = input("Enter the filename of the device map (e.g., device_map.json): ")
        if not filename.strip():
            print("Filename cannot be empty. Please try again.")
            continue
        if not filename.endswith(".json"):
            filename += ".json"
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found. Please check the path and try again.")
            continue
        break

    merged_data = {}
    try:
        with open(filename, 'r') as f:
            merged_data = json.load(f)
        print(f"Successfully loaded merged data from '{filename}'.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'. Please ensure it's a valid JSON file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading '{filename}': {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_merged_lattice(ax, merged_data)
    plt.show()

if __name__ == "__main__":
    plot_merged_layout_from_file()
