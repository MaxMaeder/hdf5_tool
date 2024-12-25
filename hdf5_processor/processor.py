import os
import sys
import h5py
import numpy as np
from collections import OrderedDict, defaultdict
from hdf5_processor.util import save_to_csv, list_hdf5_files


def process_hdf5_files(input_folder: str, output_folder: str):
    """
    Reads all HDF5 files in 'input_folder', computes:
      (1) average XYZ position per sensor per device
      (2) max Euclidean distance per sensor per device
    and writes two CSVs in 'output_folder':
      - average_positions.csv
      - max_distances.csv

    Columns are consistent across files. Each row is one file.
    """

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist.")
    if not os.path.isdir(output_folder):
        raise FileNotFoundError(f"Output folder '{output_folder}' does not exist.")

    hdf5_files = list_hdf5_files(input_folder)
    if not hdf5_files:
        raise ValueError("No valid HDF5 data files found in the input folder.")

    # We use an OrderedDict as an ordered set to maintain 
    # consistent column ordering for device-sensor pairs
    device_sensor_set = OrderedDict()

    # Dicts to store the results: dict[file_name][(device, sensor_idx)]
    average_data = defaultdict(dict)
    max_dist_data = defaultdict(dict)

    # --- First pass: read files, collect data, and discover device–sensor pairs ---
    for file_name, file_path in hdf5_files:
        try:
            with h5py.File(file_path, "r") as hdf:
                # Loop over each device in the file
                for device in hdf.keys():
                    position_data = hdf[device].get("Position")
                    if position_data is None:
                        continue

                    num_sensors = position_data.shape[1]

                    # Loop over each sensor in the device
                    for sensor_idx in range(num_sensors):
                        # Gather all samples for this sensor
                        sensor_data = position_data[:, sensor_idx, :]  # shape = [samples, 3]

                        # Calculate average position (x, y, z)
                        avg_xyz = sensor_data.mean(axis=0)  # shape = (3,)
                        # Calculate maximum Euclidean distance
                        distances = np.linalg.norm(sensor_data, axis=1)  # shape = (samples,)
                        max_dist = distances.max()

                        # Store in our dict for this file
                        average_data[file_name][(device, sensor_idx)] = tuple(avg_xyz)
                        max_dist_data[file_name][(device, sensor_idx)] = float(max_dist)

                        # Ensure (device, sensor_idx) is in the OrderedDict
                        if (device, sensor_idx) not in device_sensor_set:
                            device_sensor_set[(device, sensor_idx)] = None

        except Exception as e:
            print(f"Error processing file '{file_name}': {e}", file=sys.stderr)

    # --- Build CSV for AVERAGE positions ---
    average_header = ["FileName"]
    for device, sensor_idx in device_sensor_set.keys():
        average_header.append(f"{device}_Sensor{sensor_idx}_X")
        average_header.append(f"{device}_Sensor{sensor_idx}_Y")
        average_header.append(f"{device}_Sensor{sensor_idx}_Z")

    average_rows = [average_header]  # First row is header
    for file_name, _ in hdf5_files:
        row = [file_name]
        for device, sensor_idx in device_sensor_set.keys():
            if (device, sensor_idx) in average_data[file_name]:
                xyz = average_data[file_name][(device, sensor_idx)]  # (x, y, z)
                row.extend([xyz[0], xyz[1], xyz[2]])
            else:
                row.extend(["", "", ""])  # Missing data
        average_rows.append(row)

    save_to_csv(os.path.join(output_folder, "average_positions.csv"), average_rows)

    # --- Build CSV for MAX distances ---
    max_dist_header = ["FileName"]
    for device, sensor_idx in device_sensor_set.keys():
        max_dist_header.append(f"{device}_Sensor{sensor_idx}_Dist")

    max_dist_rows = [max_dist_header]  # First row is header
    for file_name, _ in hdf5_files:
        row = [file_name]
        for device, sensor_idx in device_sensor_set.keys():
            if (device, sensor_idx) in max_dist_data[file_name]:
                dist = max_dist_data[file_name][(device, sensor_idx)] # Max distance
                row.append(dist)
            else:
                row.append("")  # Missing data
        max_dist_rows.append(row)

    save_to_csv(os.path.join(output_folder, "max_distances.csv"), max_dist_rows)
