import json
import os

def split_device_map_json():
    print("--- JSON Device Map Splitter ---")

    # Get the merged JSON filename from the user
    while True:
        merged_filename = input("Enter the filename of the device map JSON file (e.g., device_map.json): ")
        if not merged_filename.strip():
            print("Filename cannot be empty. Please try again.")
            continue
        if not merged_filename.endswith(".json"):
            merged_filename += ".json"
        if not os.path.exists(merged_filename):
            print(f"Error: File '{merged_filename}' not found. Please check the path and try again.")
            continue
        break

    merged_data = {}
    try:
        with open(merged_filename, 'r') as f:
            merged_data = json.load(f)
        print(f"Successfully loaded merged data from '{merged_filename}'.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{merged_filename}'. Please ensure it's a valid JSON file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading '{merged_filename}': {e}")
        return

    # Define keys for the qubit JSON file
    qubit_keys = [
        "optimized_qubit_coordinates",
        "optimized_couplings"
    ]

    # Define keys for the wire (control line) JSON file
    wire_keys = [
        "fixed_drive_points",
        "optimized_multiplexer_lines",
        "optimized_readout_to_multiplexer_lines",
        "optimized_drive_paths"
    ]

    qubit_data = {}
    wire_data = {}

    # Populate qubit_data
    for key in qubit_keys:
        if key in merged_data:
            qubit_data[key] = merged_data[key]

    # Populate wire_data
    for key in wire_keys:
        if key in merged_data:
            wire_data[key] = merged_data[key]

    print("\nData successfully separated.")

    # Get the output filename for the qubit JSON
    while True:
        qubit_output_filename = input("Enter the desired filename for the qubit JSON data (e.g., split_qubit_layout.json): ")
        if not qubit_output_filename.strip():
            print("Output filename cannot be empty. Please try again.")
            continue
        if not qubit_output_filename.endswith(".json"):
            qubit_output_filename += ".json"
        break

    # Get the output filename for the wire JSON
    while True:
        wire_output_filename = input("Enter the desired filename for the wire JSON data (e.g., split_wire_layout.json): ")
        if not wire_output_filename.strip():
            print("Output filename cannot be empty. Please try again.")
            continue
        if not wire_output_filename.endswith(".json"):
            wire_output_filename += ".json"
        if qubit_output_filename == wire_output_filename:
            print("The wire JSON filename cannot be the same as the qubit JSON filename. Please enter a different filename.")
            continue
        break

    # Save the qubit data to its file
    try:
        with open(qubit_output_filename, 'w') as f:
            json.dump(qubit_data, f, indent=4)
        print(f"\nSuccessfully saved qubit data to '{qubit_output_filename}'.")
        print("-" * 50)
        print("Preview of Qubit Data:")
        print(json.dumps(qubit_data, indent=4))
        print("-" * 50)
    except IOError as e:
        print(f"Error: Could not write to file '{qubit_output_filename}'. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving qubit JSON: {e}")

    # Save the wire data to its file
    try:
        with open(wire_output_filename, 'w') as f:
            json.dump(wire_data, f, indent=4)
        print(f"\nSuccessfully saved wire data to '{wire_output_filename}'.")
        print("-" * 50)
        print("Preview of Wire Data:")
        print(json.dumps(wire_data, indent=4))
        print("-" * 50)
    except IOError as e:
        print(f"Error: Could not write to file '{wire_output_filename}'. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving wire JSON: {e}")

if __name__ == "__main__":
    split_device_map_json()
