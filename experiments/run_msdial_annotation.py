"""Repository entry point using the editable polarity-specific method snapshots."""

from dataclasses import replace
from pathlib import Path

from hrms_utils.msdial_annotation import (
    build_argument_parser,
    configuration_from_arguments,
    main,
)


if __name__ == "__main__":
    config = configuration_from_arguments(build_argument_parser().parse_args())
    if config.msdial_method is None:
        config = replace(
            config,
            msdial_method=Path(
                f"experiments/msdial_console_params_{config.polarity}.txt"
            ),
        )
    main(config)
