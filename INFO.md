# Qartographer: Your Quantum Circuit Cartographer

Qartographer is a Python-based tool designed to help you optimize the placement of qubit ancillary control devices.
## 🚀 Getting Started

Since this project consists of a single file, you don't need to install anything. Just follow these steps to get started:

1.  **Download the file:** Download `qartographer.py` and place it in your project directory, or clone this repository with `git clone https://github.com/AYDOSUL/Qartographer.git`

2.  **Import the Dependencies:** Import NumPy, SciPy, MatPlotLib, JSON, and OS.
3.  **Create Your First JSON:** Create a JSON file with the structure discussed in `Qartographer_Whitepaper.pdf`
4.  **Choose Your Drive Points:** Choose and edit the drive points array to match your chip specifications
5.  **Edit The Optimization Constants:** Edit the values to match your specifications
6.  **Run The Program!**
7.  **Save the JSON File**
## ➡️ Next Steps
Now that you have both the qubit and wiring json files, you can use the merger to combine the two JSONs into one device map. 
This map can then be exported or shared, and it contains all of the qubit location and component data. This means that the qubit data can be plotted
via the plotter. If needed, one can split the device map for further optimization or manual editing.
