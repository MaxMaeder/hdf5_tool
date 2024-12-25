import argparse
from hdf5_processor.processor import process_hdf5_files
import sys


def main():
    """
    Entry point for the HDF5 Processor command-line interface (CLI).

    This function:
      1. Parses two command-line arguments (input_folder, output_folder).
      2. Calls the process_hdf5_files(...) function to process HDF5 data files found in input_folder.
      3. Saves the resulting CSV files (average_positions.csv and max_distances.csv) in output_folder.
      4. Prints a success message if processing is completed without errors, or an error message otherwise.
    """
    
    parser = argparse.ArgumentParser(description="Process HDF5 tracking data files.")
    parser.add_argument(
        "input_folder", type=str, help="Folder containing HDF5 data files."
    )
    parser.add_argument(
        "output_folder", type=str, help="Folder to save output CSV files."
    )
    args = parser.parse_args()

    try:
        process_hdf5_files(args.input_folder, args.output_folder)
        print("Processing completed successfully.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
