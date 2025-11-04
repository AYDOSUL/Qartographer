# 🛠️ Qartographer JSONbuilder Guide: Qubit Placement Data

The **Qubit Placement Data** file is the **initial input** for the Qartographer optimization workflow. It establishes the physical location of the qubits and defines the fixed coupling architecture before ancillary components (readout and control lines) are routed.

---

## 🏗️ JSON Structure Overview

The file must be a single JSON object containing two top-level keys:

1.  `"optimized_qubit_coordinates"`: An object mapping qubit IDs to their (x, y) coordinates on the chip.
2.  `"optimized_couplings"`: An array of objects defining which qubits are physically coupled together (i.e., nearest neighbors).

```json
{
  "optimized_qubit_coordinates": {
    "Qubit_0": {
      "x": 0.000677,
      "y": -9.5e-5
    },
    ... all other qubits ...
  },
  "optimized_couplings": [
    {
      "qubit1_index": 0,
      "qubit2_index": 1
    },
    ... all coupling pairs ...
  ]
}
```

---

## 1. Defining Qubit Coordinates (`optimized_qubit_coordinates`)

This object specifies the **pre-optimized geometric position** of every qubit on your chip.

### Key Rules:

* **Key Format:** Each key must be a string following the pattern `"Qubit_N"`, where N is the integer index of the qubit (starting at **0**).
* **Coordinate Units:** The x and y values must be **floating-point numbers** and should be specified in **meters**. Scientific notation (e.g., `9.5e-5`) is acceptable.
* **Structure:** Each qubit key maps to a nested object with keys `"x"` and `"y"`.

### Example Snippet:

```json
  "optimized_qubit_coordinates": {
    "Qubit_0": {
      "x": 0.000677,  # x-coordinate (in meters)
      "y": -9.5e-5    # y-coordinate (in meters)
    },
    "Qubit_1": {
      "x": 0.000735,
      "y": -9.5e-5
    },
    ...
    "Qubit_48": {
      "x": 0.001025,
      "y": 0.000253
    }
  },
```

---

## 2. Defining Qubit Couplings (`optimized_couplings`)

This array defines the physical **connectivity map** of the quantum processor.

### Key Rules:

* **Array of Objects:** The value of `"optimized_couplings"` is an array (`[]`) containing objects (`{}`).
* **Connectivity:** Each object represents a single physical coupling (a direct connection/interaction) between two qubits.
* **Index Usage:** You must use the **integer index** of the qubits, ***not*** the full string ID (e.g., use `0` instead of `"Qubit_0"`).
* **Undirected Couplings:** The order of `qubit1_index` and `qubit2_index` does not strictly matter, but it's best practice to **list the smaller index first** (e.g., use `(0, 1)` rather than `(1, 0)`).

### Example Snippet (showing horizontal and vertical couplings for a lattice):

```json
  "optimized_couplings": [
    {
      "qubit1_index": 0,
      "qubit2_index": 1
    },
    {
      "qubit1_index": 0,
      "qubit2_index": 7
    },
    {
      "qubit1_index": 1,
      "qubit2_index": 2
    }
    ... continue listing all coupled pairs ...
  ]
```

---

## ⚠️ Checklist for Validation

Before running Qartographer, ensure your JSON file meets these requirements:

* **All Qubits Present:** Every qubit from `Qubit_0` up to `Qubit_N` must be listed under `optimized_qubit_coordinates`.
* **Consistent Indexing:** The indices used in `optimized_couplings` must correspond exactly to the indices used in `optimized_qubit_coordinates`.
* **Floating-Point Coordinates:** All `x` and `y` values must be numbers (floats).
* **No Redundant Couplings:** Do not list the same coupling pair twice (e.g., `(0, 1)` and `(1, 0)`).
* **Final Structure:** The entire JSON must be enclosed in curly braces `{}`.
