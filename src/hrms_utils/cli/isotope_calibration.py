"""Installed isotope-calibration command."""
from hrms_utils.isotope_calibration import build_argument_parser, configuration_from_arguments, main

def cli_main() -> None:
    main(configuration_from_arguments(build_argument_parser().parse_args()))

def main_cli() -> None:
    cli_main()

if __name__ == "__main__":
    cli_main()
