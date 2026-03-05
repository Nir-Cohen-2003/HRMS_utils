import argparse

from hrms_utils.formats.pubchem import process_pubchem_data


def main():
    parser = argparse.ArgumentParser(
        description="Process PubChem data for spectral library enrichment."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing PubChem files from FTP if not present",
    )
    parser.add_argument(
        "--input-dir",
        default="data/",
        help="Directory containing or for storing PubChem gz files",
    )
    parser.add_argument(
        "--output",
        default="pubchem.parquet",
        help="Output path for the merged Parquet file",
    )
    args = parser.parse_args()

    process_pubchem_data(args.input_dir, args.output, download_if_missing=args.download)


if __name__ == "__main__":
    main()
