import unittest
import os
import shutil
import tempfile
import numpy as np
import h5py
from unittest.mock import patch
from io import StringIO

import hdf5_processor.cli


class TestHDF5ProcessorCLI(unittest.TestCase):

    def setUp(self):
        """
        Create a temporary directory for input HDF5 files and output CSV files.
        """
        self.temp_input_dir = tempfile.mkdtemp(prefix="test_input_")
        self.temp_output_dir = tempfile.mkdtemp(prefix="test_output_")

    def tearDown(self):
        """
        Clean up by removing the temporary directories.
        """
        shutil.rmtree(self.temp_input_dir)
        shutil.rmtree(self.temp_output_dir)

    def _create_hdf5_file(self, file_name, device_sensor_data):
        """
        Helper method to create a single HDF5 file in self.temp_input_dir.

        device_sensor_data should be a dictionary:
            { "DeviceName": np.array(...) }
        Where the array has shape (samples, sensors, 3).
        """
        path = os.path.join(self.temp_input_dir, file_name)
        with h5py.File(path, "w") as hdf:
            for device_name, pos_data in device_sensor_data.items():
                grp = hdf.create_group(device_name)
                grp.create_dataset("Position", data=pos_data)
        return path

    def test_cli_invocation(self):
        """
        Test invoking the HDF5 processor via the CLI
        """
        # Create a sample HDF5 file
        pos_data = np.random.rand(10, 2, 3)
        self._create_hdf5_file("cli_test.hdf5", {"DeviceCLI": pos_data})

        # Fake args to CLI
        fake_argv = [
            "hdf5_processor.cli",  # script name
            self.temp_input_dir,  # input_folder
            self.temp_output_dir,  # output_folder
        ]

        buf = StringIO()
        with patch("sys.argv", fake_argv), \
            patch("sys.stdout", buf), \
            patch("sys.stderr", buf):
            # Call the CLI's main function
            hdf5_processor.cli.main()

        output = buf.getvalue()
        self.assertIn("Processing completed successfully.", output)

        # Confirm CSVs were generated
        avg_csv_path = os.path.join(self.temp_output_dir, "average_positions.csv")
        max_csv_path = os.path.join(self.temp_output_dir, "max_distances.csv")
        self.assertTrue(os.path.exists(avg_csv_path))
        self.assertTrue(os.path.exists(max_csv_path))


if __name__ == "__main__":
    unittest.main()
