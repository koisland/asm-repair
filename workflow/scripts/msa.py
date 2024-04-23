import polars as pl
import numpy as np
import bidict as bd
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
        .with_columns(mon_id=pl.concat_str([pl.col("monomer", "identity")], separator="_"))
        .collect()
    )


def main():
    df_verkko = format_strdec_data("data/verkko.tsv")
    df_hifiasm = format_strdec_data("data/hifiasm.tsv")
    df_exp = pl.read_csv(
        "data/expected.tsv",
        separator="\t",
        has_header=False,
        new_columns=["ctg", "monomer", "start", "stop", "identity", "len"],
    )
    mon_ids = pl.concat(
        [df_verkko.get_column("mon_id"), df_hifiasm.get_column("mon_id")]
    ).unique()

    # Convert each id to a single character.
    # SAFE: No duplicates. ValueErrors if above 1,114,111 limit.
    # https://docs.python.org/3/library/functions.html#chr
    assert len(mon_ids) < 1_114_111, "Too many monomer IDs"
    mid_mappings = bd.bidict(
        {mid: chr(i) if i != 45 else chr(1_114_111) for i, mid in enumerate(mon_ids)}
    )

    dfs = [
        df_verkko.with_columns(pl.col("mon_id").replace(mid_mappings)),
        df_hifiasm.with_columns(pl.col("mon_id").replace(mid_mappings))
    ]
    seqs = [
        "".join(df.get_column("mon_id").to_list())
        for df in dfs
    ]

    mapping_alphabet = seq.Alphabet(mid_mappings.values())
    # Penalize mismatches.
    matrix = align.SubstitutionMatrix(
        mapping_alphabet, mapping_alphabet, np.identity(len(mapping_alphabet))
    )

    ali, order, tree, dst_mtx = align.align_multiple(
        [
            seq.GeneralSequence(alphabet=mapping_alphabet, sequence=s)
            for s in seqs
        ],
        matrix,
        # Don't penalize gaps.
        gap_penalty=(0, 0),
        terminal_penalty=False,
    )
    concensus_rows = []
    for trace in ali.trace:
        for i, seq_idx in enumerate(trace):
            if seq_idx != -1:
                row = dfs[i].row(seq_idx)
                # TODO: Check for misassembly.
                concensus_rows.append(row)
                break
            else:
                continue
    
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
            breaks=pl.col("ctg").rle_id()
        ))
    break_map = dict(df_concensus_breaks.select("breaks", "ctg").unique().iter_rows())
    df_summary = (
        df_concensus_breaks
        .group_by("breaks")
        .agg(
            start=pl.col("start").min(),
            stop=pl.col("stop").max(),
            ort=pl.col("mon").str.contains("'")
        )
        .with_columns(
            ctg=pl.col("breaks").replace(break_map)
        )
        .drop("breaks")
        .sort(by="start")
        .select("ctg", "start", "stop", "ort")
    )
    breakpoint()



if __name__ == "__main__":
    raise SystemExit(main())
