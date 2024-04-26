import sys
import argparse
import polars as pl
import numpy as np
import biotite.sequence as seq
import biotite.sequence.align as align


def format_strdec_data(input_file: str) -> pl.DataFrame:
    return (
        pl.scan_csv(
            input_file,
            separator="\t",
            has_header=False,
            new_columns=["ctg", "monomer", "start", "stop", "identity"],
        )
        .with_columns(
            ctg_name_coords=pl.col("ctg").str.split_exact(by=":", n=1)
        )
        .unnest("ctg_name_coords")
        .rename({"field_0": "ctg_name", "field_1": "coords"})
        .with_columns(
            coord_pair=pl.col("coords").str.split_exact(by="-", n=1)
        )
        .unnest("coord_pair")
        .rename({"field_0": "ctg_start", "field_1": "ctg_stop"})
        .cast({"ctg_start": pl.Int64, "ctg_stop": pl.Int64})
        .with_columns(
            # TODO: Check if this is right if aligned in - ort.
            # start=pl.col("start") + pl.col("ctg_start"),
            # stop=pl.col("stop") + pl.col("ctg_start"),
            mon_id=pl.concat_str([pl.col("monomer", "identity")], separator="_")
        )
        .select("ctg", "monomer", "start", "stop", "identity", "mon_id")
        .collect()
    )


def main():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("-i", "--infiles", nargs="+", help="Input StringDecomposer outputs.")
    ap.add_argument("-s", "--summary", default="summary.tsv", help="Coordinate summary")
    ap.add_argument("-o", "--outfile", default=sys.stdout, type=argparse.FileType("wt"), help="Consensus StringDecomposer output.")

    args = ap.parse_args()

    dfs = [format_strdec_data(f) for f in args.infiles]

    mon_ids = pl.concat(
        [df.get_column("mon_id") for df in dfs]
    ).unique()

    # Convert each id to a single character.
    # SAFE: No duplicates. ValueErrors if above 1,114,111 limit.
    # https://docs.python.org/3/library/functions.html#chr
    assert len(mon_ids) < 1_114_111, "Too many monomer IDs"
    mid_mappings = {mid: chr(i) if i != 45 else chr(1_114_111) for i, mid in enumerate(mon_ids)}

    dfs_mod = [
        df.with_columns(pl.col("mon_id").replace(mid_mappings))
        for df in dfs
    ]
    seqs = [
        "".join(df.get_column("mon_id").to_list())
        for df in dfs_mod
    ]

    mapping_alphabet = seq.Alphabet(mid_mappings.values())
    # Penalize mismatches.
    matrix = align.SubstitutionMatrix(
        mapping_alphabet, mapping_alphabet, np.identity(len(mapping_alphabet))
    )

    aln, order, tree, dst_mtx = align.align_multiple(
        [
            seq.GeneralSequence(alphabet=mapping_alphabet, sequence=s)
            for s in seqs
        ],
        matrix,
        # Heavily penalize opening multiple gaps. Allow extensions. Try to keep contigs together.
        gap_penalty=(-5, 0),
        terminal_penalty=False,
    )
    concensus_rows = []

    seq_i = 0
    for trace in aln.trace:
        matches = np.where(trace != -1)[0]
        seq_i_idx = np.where(matches == seq_i)[0]
        
        # Check if current i.
        if len(seq_i_idx) != 0:
            seq_i = seq_i_idx[0]
        # Just take first one
        elif len(matches) != 0:
            seq_i = matches[0]
        else:
            continue
        try:
            trace_idx = trace[seq_i]
        except IndexError:
            breakpoint()
        row = dfs[seq_i].row(trace_idx)
        # TODO: Check for misassembly.
        concensus_rows.append(row)

    df_concensus = (
        pl.LazyFrame(
            concensus_rows,
            schema=["ctg", "mon", "start", "stop", "identity", "mid"]
        )
        .drop("mid")
        .collect()
    )

    df_concensus_breaks = (df_concensus
        .with_columns(
            breaks=pl.col("ctg").rle_id(),
            # ort=pl.when(pl.col("mon").str.contains("'")).then(pl.lit("-")).otherwise(pl.lit("+"))
        ))
    break_map = dict(df_concensus_breaks.select("breaks", "ctg").unique().iter_rows())
    # TODO: Add orientation
    df_summary = (
        df_concensus_breaks
        .group_by("breaks")
        .agg(
            start=pl.col("start").min(),
            stop=pl.col("stop").max(),
        )
        .with_columns(
            ctg=pl.col("breaks").replace(break_map),
            length=pl.col("stop")-pl.col("start")
        )
        .sort(by="breaks")
        .drop("breaks")
        .with_columns(
            asm_start=pl.col("length").cum_sum().shift(1).fill_null(0),
            asm_stop=pl.col("length").cum_sum()
        )
        .select("ctg", "start", "stop", "asm_start", "asm_stop", "length")
    )

    df_summary.write_csv(
        args.summary, separator="\t", include_header=True
    )
    df_concensus.write_csv(
        args.outfile, separator="\t", include_header=False
    )


if __name__ == "__main__":
    raise SystemExit(main())
