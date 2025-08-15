import json
import os

def merge_json_files():
    print("--- JSON File Merger ---")

    # Get the first JSON filename from the user
    while True:
        file1_name = input("Enter the name of the file containing the qubit layout: ")
        if not file1_name.strip():
            print("Filename cannot be empty. Please try again.")
            continue
        if not file1_name.endswith(".json"):
            file1_name += ".json"
        if not os.path.exists(file1_name):
            print(f"Error: File '{file1_name}' not found. Please check the path and try again.")
            continue
        break

    # Get the second JSON filename from the user
    while True:
        file2_name = input("Enter the name of the file containing the wiring layout: ")
        if not file2_name.strip():
            print("Filename cannot be empty. Please try again.")
            continue
        if not file2_name.endswith(".json"):
            file2_name += ".json"
        if not os.path.exists(file2_name):
            print(f"Error: File '{file2_name}' not found. Please check the path and try again.")
            continue
        if file1_name == file2_name:
            print("The second filename cannot be the same as the first. Please enter a different filename.")
            continue
        break

    data1 = {}
    data2 = {}

    # Load data from the first file
    try:
        with open(file1_name, 'r') as f:
            data1 = json.load(f)
        print(f"Successfully loaded data from '{file1_name}'.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file1_name}'. Please ensure it's a valid JSON file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file1_name}': {e}")
        return

    # Load data from the second file
    try:
        with open(file2_name, 'r') as f:
            data2 = json.load(f)
        print(f"Successfully loaded data from '{file2_name}'.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file2_name}'. Please ensure it's a valid JSON file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file2_name}': {e}")
        return
    merged_data = {**data1, **data2}
    print("\nFiles merged successfully.")

    # Get the output filename from the user
    while True:
        output_filename = input("Enter the desired filename for the device map (e.g., device_map.json): ")
        if not output_filename.strip():
            print("Output filename cannot be empty. Please try again.")
            continue
        if not output_filename.endswith(".json"):
            output_filename += ".json"
        break

    # Save the merged data to the new file
    try:
        with open(output_filename, 'w') as f:
            json.dump(merged_data, f, indent=4)
        print(f"\nSuccessfully saved merged data to '{output_filename}'.")
        print("-" * 50)
        print("Preview of Merged Data:")
        print(json.dumps(merged_data, indent=4))
        print("-" * 50)
    except IOError as e:
        print(f"Error: Could not write to file '{output_filename}'. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving JSON: {e}")

if __name__ == "__main__":
    merge_json_files()
