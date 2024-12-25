import unittest
import os
import csv
import tempfile
import shutil

from hdf5_processor.util import save_to_csv, list_hdf5_files

class TestUtils(unittest.TestCase):
    def setUp(self):
        """
        Create a temporary directory for testing.
        """
        self.test_dir = tempfile.mkdtemp(prefix="test_utils_")

    def tearDown(self):
        """
        Remove the temporary directory after each test.
        """
        shutil.rmtree(self.test_dir)

    def test_save_to_csv(self):
        """
        Test that save_to_csv writes rows correctly.
        """
        test_data = [
            ["Header1", "Header2", "Header3"],
            [1, 2, 3],
            ["A", "B", "C"]
        ]
        csv_path = os.path.join(self.test_dir, "test_output.csv")

        # Write CSV
        save_to_csv(csv_path, test_data)

        # Verify that the file was created
        self.assertTrue(os.path.exists(csv_path))

        # Read back the CSV and check content
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.assertEqual(rows, [
            ["Header1", "Header2", "Header3"],
            ["1", "2", "3"],
            ["A", "B", "C"]
        ])

    def test_list_hdf5_files(self):
        """
        Test that list_hdf5_files returns only .hdf5 files in sorted order.
        """
        # Create a mix of files
        filenames = [
            "data1.hdf5",
            "data2.txt",
            "another_data.hdf5",
            "somefile.jpg"
        ]
        for fname in filenames:
            file_path = os.path.join(self.test_dir, fname)
            with open(file_path, "w") as f:
                f.write("mock content")

        # The function returns a list of (filename, full_path) for only .hdf5 files
        hdf5_files = list_hdf5_files(self.test_dir)
        # Sort by the first element in each tuple for easier checking
        self.assertEqual(
            hdf5_files,
            [
                ("another_data.hdf5", os.path.join(self.test_dir, "another_data.hdf5")),
                ("data1.hdf5",        os.path.join(self.test_dir, "data1.hdf5")),
            ]
        )

if __name__ == "__main__":
    unittest.main()
