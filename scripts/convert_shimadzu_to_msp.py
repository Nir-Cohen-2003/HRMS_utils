"""Convert Shimadzu spectral library export to MSP format."""

import argparse
import sys
from pathlib import Path

import polars as pl


def read_shimadzu(path: Path) -> pl.DataFrame:
    """Read a Shimadzu CSV export and return a tidy DataFrame."""
    raw = pl.read_csv(
        path,
        comment_prefix=None,
        infer_schema_length=0,
    )

    # Collect fragment pairs into a list of (m/z, intensity) per row.
    max_frags = 7
    frag_pairs = []
    for i in range(max_frags):
        col_mz = f"m/z {i + 1}"
        col_int = f"Intensity {i + 1}"
        if col_mz in raw.columns and col_int in raw.columns:
            frag_pairs.append(
                pl.struct(
                    [
                        pl.col(col_mz).cast(pl.Float64, strict=False).alias("mz"),
                        pl.col(col_int)
                        .cast(pl.Float64, strict=False)
                        .alias("intensity"),
                    ]
                )
            )

    if frag_pairs:
        df = raw.with_columns(pl.concat_list(frag_pairs).alias("fragments_raw"))
    else:
        df = raw.with_columns(pl.lit([]).alias("fragments_raw"))

    # Drop empty / None fragments while preserving order.
    df = df.with_columns(
        pl.col("fragments_raw")
        .list.eval(
            pl.when(
                pl.element().struct.field("mz").is_not_null()
                & pl.element().struct.field("intensity").is_not_null()
            ).then(pl.element())
        )
        .list.drop_nulls()
        .alias("fragments")
    ).drop("fragments_raw")

    return df


def _map_ion_mode(mode: str | None) -> str:
    """Map full polarity string to single-letter p/n used by MSP parsers."""
    if mode is None:
        return ""
    mode_lower = mode.lower()
    if mode_lower == "positive":
        return "p"
    if mode_lower == "negative":
        return "n"
    return mode_lower


def write_msp(df: pl.DataFrame, out_path: Path) -> None:
    """Write a DataFrame in MSP format compatible with nist_mspec.py."""
    with open(out_path, "w") as fh:
        for row in df.iter_rows(named=True):
            name = row.get("Compound Name", "")
            exact_mass = row.get("Theory MW", "")
            formula = row.get("Formula", "")
            precursor_mz = row.get("Precursor m/z", "")
            precursor_type = row.get("Precursor Ion", "")
            smiles = row.get("SMILES", "")
            inchi = row.get("InChI", "")
            inchikey = row.get(
                "Comment", ""
            )  # header says Comment, content is InChIKey
            ion_mode = _map_ion_mode(row.get("Polarity", ""))
            fragments: list[dict] = row.get("fragments", [])
            num_peaks = len(fragments)

            fh.write(f"Name: {name}\n")
            fh.write(f"ExactMass: {exact_mass}\n")
            fh.write(f"Formula: {formula}\n")
            fh.write(f"PrecursorMZ: {precursor_mz}\n")
            fh.write(f"Precursor_type: {precursor_type}\n")
            fh.write(f"Ion_mode: {ion_mode}\n")
            fh.write(f"SMILES: {smiles}\n")
            fh.write(f"InChI: {inchi}\n")
            fh.write(f"InChIKey: {inchikey}\n")
            fh.write(f"Num Peaks: {num_peaks}\n")

            # Write peaks in m/z <space> intensity format.
            for frag in fragments:
                mz = frag["mz"]
                intensity = frag["intensity"]
                fh.write(f"{mz} {intensity}\n")

            # Consistent blank-line separator between spectra.
            fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Shimadzu CSV spectral library to MSP format."
    )
    parser.add_argument("input", type=Path, help="Path to Shimadzu CSV file.")
    parser.add_argument("output", type=Path, help="Path to output MSP file.")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    df = read_shimadzu(args.input)
    write_msp(df, args.output)
    print(f"Wrote {len(df)} spectra to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
