import unittest
import os
import shutil
import tempfile
import numpy as np
import h5py

from hdf5_processor.processor import process_hdf5_files


class TestHDF5Processor(unittest.TestCase):

    def setUp(self):
        """
        Create a temporary directory for input HDF5 files and
        another for output CSV files before each test.
        """
        self.temp_input_dir = tempfile.mkdtemp(prefix="test_input_")
        self.temp_output_dir = tempfile.mkdtemp(prefix="test_output_")

    def tearDown(self):
        """
        Remove the temporary input and output directories after each test.
        """
        shutil.rmtree(self.temp_input_dir)
        shutil.rmtree(self.temp_output_dir)

    def _create_hdf5_file(self, file_name, device_sensor_data):
        """
        Helper method that creates a single HDF5 file in the input directory,
        based on a dictionary mapping of
            device_name -> position_data (numpy array, shape = (samples, sensors, 3))

        device_sensor_data example:
        {
          "DeviceA": np.random.rand(100, 2, 3),
          "DeviceB": np.random.rand(50, 1, 3)
        }
        """
        path = os.path.join(self.temp_input_dir, file_name)
        with h5py.File(path, "w") as hdf:
            for device_name, pos_data in device_sensor_data.items():
                grp = hdf.create_group(device_name)
                grp.create_dataset("Position", data=pos_data)
        return path

    def test_single_device_single_sensor(self):
        """
        Test processing with a single device that has a single sensor.
        We expect the average_positions CSV to have columns:
            [FileName, DeviceA_Sensor0_X, DeviceA_Sensor0_Y, DeviceA_Sensor0_Z]
        We expect the max_distances CSV to have columns:
            [FileName, DeviceA_Sensor0_Dist]
        """
        # Create a file with 1 device, 1 sensor, 10 samples
        pos_data = np.array(
            [[[x, x + 1, x + 2]] for x in range(10)]
        )  # shape = (10, 1, 3)
        # pos_data[:, 0, :] ranges from [[0,1,2], [1,2,3], ... [9,10,11]]

        device_data = {"DeviceA": pos_data}
        self._create_hdf5_file("test_single.hdf5", device_data)

        process_hdf5_files(self.temp_input_dir, self.temp_output_dir)

        # Check average_positions.csv
        avg_csv = os.path.join(self.temp_output_dir, "average_positions.csv")
        self.assertTrue(os.path.exists(avg_csv))

        with open(avg_csv, "r") as f:
            lines = f.read().strip().split("\n")
        # Header should be 1 line, data rows for each file is 1 line
        self.assertEqual(len(lines), 2)

        header = lines[0].split(",")
        data_row = lines[1].split(",")
        self.assertEqual(
            header,
            ["FileName", "DeviceA_Sensor0_X", "DeviceA_Sensor0_Y", "DeviceA_Sensor0_Z"],
        )
        self.assertEqual(data_row[0], "test_single.hdf5")

        # The average across sensor_data for x, y, z
        # sensor_data is: [[0,1,2], [1,2,3], ..., [9,10,11]]
        # mean for x is 4.5, for y is 5.5, for z is 6.5
        self.assertAlmostEqual(float(data_row[1]), 4.5, delta=1e-6)
        self.assertAlmostEqual(float(data_row[2]), 5.5, delta=1e-6)
        self.assertAlmostEqual(float(data_row[3]), 6.5, delta=1e-6)

        # Check max_distances.csv
        max_csv = os.path.join(self.temp_output_dir, "max_distances.csv")
        self.assertTrue(os.path.exists(max_csv))

        with open(max_csv, "r") as f:
            lines = f.read().strip().split("\n")
        self.assertEqual(len(lines), 2)

        header = lines[0].split(",")
        data_row = lines[1].split(",")
        self.assertEqual(header, ["FileName", "DeviceA_Sensor0_Dist"])
        # max distance from origin among [ [0,1,2], [1,2,3], ..., [9,10,11] ]
        # The last sample is [9,10,11], so Euclidean norm = sqrt(9^2 + 10^2 + 11^2) = sqrt(81 + 100 + 121) = sqrt(302)
        self.assertAlmostEqual(float(data_row[1]), np.sqrt(302), delta=1e-6)

    def test_multiple_devices_sensors_overlap(self):
        """
        Test multiple files, each having multiple devices and sensors,
        some overlapping, some not. Ensures columns line up properly.
        """
        # File 1: DeviceA has 2 sensors, DeviceB has 1 sensor
        pos_data_A = np.random.rand(100, 2, 3)  # shape = (100, 2, 3)
        pos_data_B = np.random.rand(100, 1, 3)  # shape = (100, 1, 3)
        self._create_hdf5_file(
            "multi1.hdf5", {"DeviceA": pos_data_A, "DeviceB": pos_data_B}
        )

        # File 2: DeviceA has 1 sensor, DeviceC has 2 sensors
        pos_data_A2 = np.random.rand(50, 1, 3)  # shape = (50, 1, 3)
        pos_data_C = np.random.rand(50, 2, 3)  # shape = (50, 2, 3)
        self._create_hdf5_file(
            "multi2.hdf5", {"DeviceA": pos_data_A2, "DeviceC": pos_data_C}
        )

        process_hdf5_files(self.temp_input_dir, self.temp_output_dir)

        # Now read the "average_positions.csv" and see how many columns we have
        avg_csv = os.path.join(self.temp_output_dir, "average_positions.csv")
        with open(avg_csv, "r") as f:
            lines = f.read().strip().split("\n")

        header = lines[0].split(",")
        # We have 2 files (multi1.hdf5, multi2.hdf5) => 2 data rows + 1 header => total 3 lines
        self.assertEqual(len(lines), 3)

        # The devices and sensors we have:
        # multi1.hdf5: (DeviceA, Sensor0), (DeviceA, Sensor1), (DeviceB, Sensor0)
        # multi2.hdf5: (DeviceA, Sensor0), (DeviceC, Sensor0), (DeviceC, Sensor1)
        # => In total:
        #   (DeviceA, Sensor0), (DeviceA, Sensor1),
        #   (DeviceB, Sensor0), (DeviceC, Sensor0), (DeviceC, Sensor1)
        # For each sensor, we have 3 columns in average_positions => 5 sensor pairs * 3 + 1 "FileName" column = 16
        self.assertEqual(
            len(header), 1 + 5 * 3
        )  # 1 for FileName, plus 3 columns per sensor

        # The order is determined by the processor code, but we can check presence:
        self.assertIn("DeviceA_Sensor0_X", header)
        self.assertIn("DeviceA_Sensor1_X", header)
        self.assertIn("DeviceB_Sensor0_X", header)
        self.assertIn("DeviceC_Sensor0_X", header)
        self.assertIn("DeviceC_Sensor1_X", header)

        # Check max_distances.csv
        max_csv = os.path.join(self.temp_output_dir, "max_distances.csv")
        with open(max_csv, "r") as f:
            lines = f.read().strip().split("\n")
        header = lines[0].split(",")
        # We have 1 column per sensor => 5 sensors + 1 "FileName" = 6 columns
        self.assertEqual(len(header), 1 + 5)

    def test_large_data_sets(self):
        """
        Create a "large" dataset to ensure that we can handle it without errors.
        """
        samples = 10_000
        sensors = 5

        # Create a single HDF5 file with many samples and multiple sensors
        pos_data_large = np.random.rand(samples, sensors, 3)
        self._create_hdf5_file("large.hdf5", {"DeviceLarge": pos_data_large})

        process_hdf5_files(self.temp_input_dir, self.temp_output_dir)

        # Check output CSVs exist
        avg_csv = os.path.join(self.temp_output_dir, "average_positions.csv")
        max_csv = os.path.join(self.temp_output_dir, "max_distances.csv")
        self.assertTrue(os.path.exists(avg_csv))
        self.assertTrue(os.path.exists(max_csv))

        # Check line count (1 row for header, 1 row for data)
        with open(avg_csv, "r") as f:
            lines = f.read().strip().split("\n")
        self.assertEqual(len(lines), 2)

        # Check we have correct number of columns
        # Should have 1 + (sensors * 3) columns = 1 + 15 = 16
        data_row = lines[1].split(",")
        self.assertEqual(len(data_row), 1 + sensors * 3)


if __name__ == "__main__":
    unittest.main()
