"""Installed required-polarity MS-DIAL annotation command."""

from hrms_utils.msdial_annotation import (
    build_argument_parser,
    configuration_from_arguments,
    main,
)


def cli_main() -> None:
    main(configuration_from_arguments(build_argument_parser().parse_args()))


if __name__ == "__main__":
    cli_main()
