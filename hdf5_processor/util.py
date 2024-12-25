import csv
import os


def save_to_csv(file_path: str, data: list[list]) -> None:
    """
    Writes a list of rows to a CSV file.

    Each element of `data` is expected to be a list, representing one row in the CSV.
    For example:
        data = [
            ["Header1", "Header2", "Header3"],
            [1, 2, 3],
            ["A", "B", "C"]
        ]
    """

    with open(file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)


def list_hdf5_files(folder_path: str) -> list[str]:
    """
    Returns a sorted list of (filename, full_path) tuples for all `.hdf5` files 
    in the given folder.
    """

    hdf5_files = [
        (file, os.path.join(folder_path, file))
        for file in os.listdir(folder_path)
        if file.endswith(".hdf5")
    ]
    return sorted(hdf5_files)
