import sys
import pprint
import logging
import argparse
import polars as pl
import numpy as np
import biotite.sequence as seq
import biotite.sequence.align as align

from typing import Iterator
from collections import defaultdict

from intervaltree import IntervalTree, Interval

DEF_STRDEC_COLS = [
    "cen",
    "monomer",
    "start",
    "stop",
    "identity",
    "sec_monomer",
    "sec_identity",
    "homo_monomer",
    "homo_identity",
    "homo_sec_monomer",
    "homo_sec_identity",
    "reliability",
]

DEF_MISASM_COLS = ["ctg", "start", "stop", "type"]
DEF_SEQ_IDEN_THR = 85
DEF_NUM_MONS_BELOW_SEQ_IDEN_THR = 5

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler and set level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatter to console handler
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)


def read_misassemblies(
    input_file: str, *, contig: str, orientation: str
) -> Iterator[Interval]:
    try:
        df = (
            pl.read_csv(
                input_file, has_header=False, new_columns=DEF_MISASM_COLS, separator="\t"
            )
            .with_columns(
                ctg_name_coords=pl.col("ctg").str.split_exact(by=":", n=1),
            )
            .unnest("ctg_name_coords")
            .rename({"field_0": "ctg_name", "field_1": "coords"})
            .with_columns(coord_pair=pl.col("coords").str.split_exact(by="-", n=1))
            .unnest("coord_pair")
            .rename({"field_0": "ctg_start", "field_1": "ctg_stop"})
            .cast({"ctg_start": pl.Int64, "ctg_stop": pl.Int64})
        )
    except pl.exceptions.NoDataError:
        return None

    for m in df.filter(pl.col("ctg") == contig).iter_rows(named=True):
        if orientation == "+":
            yield Interval(m["start"] - m["ctg_start"], m["stop"] - m["ctg_start"])
        else:
            yield Interval(m["ctg_stop"] - m["stop"], m["ctg_stop"] - m["start"])


def format_strdec_data(
    input_file: str, *, orientation: str, contig_length: int
) -> pl.DataFrame:
    # Try full format and truncated format.
    try:
        lf = pl.scan_csv(
            input_file, separator="\t", has_header=False, new_columns=DEF_STRDEC_COLS
        )
    except pl.exceptions.ShapeError:
        lf = pl.scan_csv(
            input_file,
            separator="\t",
            has_header=False,
            new_columns=DEF_STRDEC_COLS[0:5],
        )
    return (
        lf
        # Extract centromere coordinates from contig name.
        .with_columns(
            cen_name_coords=pl.col("cen").str.split_exact(by=":", n=1),
            orientation=pl.lit(orientation),
            ctg_length=pl.lit(contig_length),
        )
        .unnest("cen_name_coords")
        .rename({"field_0": "cen_name", "field_1": "coords"})
        .with_columns(coord_pair=pl.col("coords").str.split_exact(by="-", n=1))
        .unnest("coord_pair")
        .rename({"field_0": "cen_start", "field_1": "cen_stop"})
        .cast({"cen_start": pl.Int64, "cen_stop": pl.Int64})
        # Merge monomer classification and its estimated sequence identity into an id.
        .with_columns(
            mon_id=pl.concat_str([pl.col("monomer", "identity")], separator="_"),
        )
        .select(
            "cen",
            "cen_start",
            "cen_stop",
            "orientation",
            "monomer",
            "start",
            "stop",
            "identity",
            "mon_id",
            "ctg_length",
        )
        .collect()
    )


def main():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument(
        "-i",
        "--infiles",
        help="Input StringDecomposer outputs as tsv file. Must contain path and orientation of contig. Each contig should also be labeled with start and stop coordinates.",
    )
    ap.add_argument(
        "-o",
        "--outfile",
        default=sys.stdout,
        type=argparse.FileType("wt"),
        help="Aligned and corrected sequence coordinates.",
    )
    ap.add_argument(
        "-c", "--consensus", default=None, help="Concensus StringDecomposer output."
    )
    ap.add_argument(
        "--base_group",
        help="Group label of contigs to prefer if multiple matches. Must correspond to a value in --infiles.",
        default=None,
    )
    ap.add_argument(
        "--gap_open_penalty",
        help="Gap opening penalty for alignment.",
        type=int,
        default=-50,
    )
    ap.add_argument(
        "--gap_ext_penalty",
        help="Gap extension penalty for alignment.",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--seq_iden_thr",
        help="Sequence identity threshold for filtering monomers. Low values signal the end of the centromere.",
        type=int,
        default=85,
    )
    ap.add_argument(
        "--num_mons_below_seq_iden_thr",
        help="Number of monomers below seq_iden_thr required to stop building consensus.",
        type=int,
        default=5,
    )

    args = ap.parse_args()

    logger.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Read input file with filename and metadata.
    # Group is string to allow arbitrary name.
    df_cen_files = pl.read_csv(
        args.infiles,
        separator="\t",
        has_header=True,
        new_columns=["ctg", "strdec", "misasm", "group", "ort", "ctg_len"],
        dtypes={"group": pl.String},
    )
    dfs: list[pl.DataFrame] = []
    grp_idxs = defaultdict(list)

    misassemblies = defaultdict(IntervalTree)
    for i, f in enumerate(df_cen_files.iter_rows(named=True)):
        df = format_strdec_data(
            f["strdec"], orientation=f["ort"], contig_length=f["ctg_len"]
        )
        dfs.append(df)
        # Store misassemblies by contig.
        for misasm in read_misassemblies(f["misasm"], contig=f["ctg"], orientation=f["ort"]):
            misassemblies[f["ctg"]].add(misasm)

        grp_idxs[f["group"]].append(i)

    base_group = args.base_group if args.base_group else next(iter(grp_idxs))
    assert (
        base_group in grp_idxs
    ), f"Specified base group ('{base_group}') is invalid. Expected: {list(grp_idxs.keys())}"
    base_group_idxs = grp_idxs[base_group]

    mon_ids = pl.concat([df.get_column("mon_id") for df in dfs]).unique()

    # Convert each id to a single character.
    # SAFE: No duplicates. ValueErrors if above 1,114,111 limit.
    # https://docs.python.org/3/library/functions.html#chr
    assert len(mon_ids) < 1_114_111, "Too many monomer IDs"
    mid_mappings = {
        mid: chr(i) if i != 45 else chr(1_114_111) for i, mid in enumerate(mon_ids)
    }

    dfs_mod = [df.with_columns(pl.col("mon_id").replace(mid_mappings)) for df in dfs]
    seqs = ["".join(df.get_column("mon_id").to_list()) for df in dfs_mod]

    mapping_alphabet = seq.Alphabet(mid_mappings.values())
    # Penalize mismatches.
    matrix = align.SubstitutionMatrix(
        mapping_alphabet, mapping_alphabet, np.identity(len(mapping_alphabet))
    )
    logger.info(f"Performing a MSA of {len(seqs)} StringDecomposer monomer sequences...")
    # https://www.biotite-python.org/apidoc/biotite.sequence.align.align_multiple.html#biotite.sequence.align.align_multiple
    aln, order, tree, dst_mtx = align.align_multiple(
        [seq.GeneralSequence(alphabet=mapping_alphabet, sequence=s) for s in seqs],
        matrix,
        # Heavily penalize opening multiple gaps. Allow extensions. Try to keep contigs together.
        gap_penalty=(args.gap_open_penalty, args.gap_ext_penalty),
        terminal_penalty=False,
    )
    concensus_rows = []
    num_mons_below_thresh = 0
    
    for trace in aln.trace:
        matches = np.where(trace != -1)[0]
        filtered_matches = []
        # Filter monomers that are in misassembled regions.
        for m in matches:
            m: int
            row = dfs[m].row(trace[m])

            # Check for overlap with misassembled region.
            if misassemblies[row[0]].overlaps(row[5], row[6]):
                continue
            filtered_matches.append(m)

        # Check if any base group in correctly assembled monomers.
        seq_i_idx = np.where(np.isin(filtered_matches, base_group_idxs))[0]
        if len(seq_i_idx) != 0:
            new_seq_i = filtered_matches[seq_i_idx[0]]
        # Just take first non-misassembled monomer if base group monomer not found.
        elif len(filtered_matches) != 0:
            new_seq_i = filtered_matches[0]
        # If no correctly assembled monomer found, don't repair.
        elif len(matches) != 0:
            new_seq_i = matches[0]
        else:
            continue

        trace_idx = trace[new_seq_i]
        row = dfs[new_seq_i].row(trace_idx)

        # Check row doesn't fall below sequence identity threshold.
        if row[7] < args.seq_iden_thr:
            num_mons_below_thresh += 1
        # Stop building consensus if number of monomers below sequence identity threshold met.
        if num_mons_below_thresh == args.num_mons_below_seq_iden_thr:
            break

        concensus_rows.append(row)

    df_concensus = (
        pl.LazyFrame(
            concensus_rows,
            schema=[
                "cen",
                "cen_start",
                "cen_stop",
                "orientation",
                "mon",
                "start",
                "stop",
                "identity",
                "mid",
                "ctg_length",
            ],
        )
        .drop("mid")
        .collect()
    )

    df_concensus_breaks = df_concensus.with_columns(
        breaks=pl.col("cen").rle_id(),
    )
    break_map = dict(df_concensus_breaks.select("breaks", "cen").unique().iter_rows())

    df_summary = (
        df_concensus_breaks.group_by(
            "breaks", "cen_start", "cen_stop", "orientation", "ctg_length"
        )
        .agg(
            start=pl.col("start").min(),
            stop=pl.col("stop").max(),
        )
        .with_columns(
            # Replace breaks with contig name.
            cen=pl.col("breaks").replace(break_map),
            cen_length=pl.col("cen_stop") - pl.col("cen_start"),
        )
        .sort(by="breaks")
        .drop("breaks")
        .with_row_index()
    )
        
    df_summary = df_summary.with_columns(
        start=pl.when(pl.col("index") == 0)
        .then(pl.lit(0))
        .otherwise(
            pl.when(pl.col("orientation") == "-")
            .then(pl.col("ctg_length") - pl.col("cen_stop") + pl.col("start"))
            .otherwise(pl.col("start") + pl.col("cen_start"))
        ),
        stop=pl.when(pl.col("index") == df_summary.shape[0] - 1)
        .then(pl.col("ctg_length"))
        .otherwise(
            pl.when(pl.col("orientation") == "-")        
            .then(pl.col("ctg_length") - pl.col("cen_stop") + pl.col("stop"))
            .otherwise(pl.col("stop") + pl.col("cen_start"))
        )
    )

    df_summary = (
        df_summary.with_columns(
            length=pl.col("stop") - pl.col("start"),
        )
        .with_columns(
            asm_start=pl.col("length").cum_sum().shift(1).fill_null(0),
            asm_stop=pl.col("length").cum_sum(),
        )
        .select(
            "cen", "start", "stop", "orientation", "asm_start", "asm_stop", "ctg_length"
        )
    )

    df_summary.write_csv(args.outfile, separator="\t", include_header=True)
    if args.consensus:
        df_concensus.write_csv(args.consensus, separator="\t", include_header=False)


if __name__ == "__main__":
    raise SystemExit(main())
