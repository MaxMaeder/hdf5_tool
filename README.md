# HDF5 Processor CLI

A Python CLI program that processes HDF5 files (containing 3D position data) and generates two CSV reports:

1. `average_positions.csv` — Average X, Y, Z position for each sensor on each device.
2. `max_distances.csv` — Maximum Euclidean distance for each sensor on each device.

## HDF5 Input Files
- Each HDF5 file should contain one or more groups (representing devices).
- Each device group contains a "Position" dataset of shape `[samples, sensors, 3]`, where:
  - samples = number of recorded positions
  - sensors = number of sensors for that device
  - 3 = the 3D coordinates (X, Y, Z)
- Your input folder should contain only the HDF5 files you wish to process (with a `.hdf5` extension).

## CSV Output Structure
The program generates two CSV files in the specified output folder:

1. `average_positions.csv`
  - Row: One row per HDF5 file.
  - Columns:
    - First column: FileName (the HDF5 file name).
    - Three columns per sensor on each device: DeviceName_SensorN_X, DeviceName_SensorN_Y, DeviceName_SensorN_Z.
    - If a device/sensor is missing in a particular file, the corresponding columns will be blank in that row.
2. `max_distances.csv`
  - Row: One row per HDF5 file.
  - Columns:
    - First column: `FileName`.
    - One column per sensor on each device: `DeviceName_SensorN_Dist`.
    - If a device/sensor is missing in a particular file, the corresponding column will be blank in that row.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hdf5-processor.git
cd hdf5-processor
```

2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the CLI module with:
```bash
python -m hdf5_processor.cli <input_folder> <output_folder>
```

- `<input_folder>`: Path to the folder containing your `.hdf5` data files.
- `<output_folder>`: Path to a folder where the CSV output files (`average_positions.csv` and `max_distances.csv`) will be written.

Try it out using test data:
```bash
python -m hdf5_processor.cli test_files test_out
```

## Testing
To run the test suite, use:
```bash
python -m unittest discover tests
```

This command discovers all tests in the `tests` directory and executes them. These test HDF5 processing logic, CSV output, and the command-line interface.