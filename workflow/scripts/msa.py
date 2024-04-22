import pyabpoa as pa
import polars as pl


def format_strdec_data(input_file: str) -> pl.DataFrame:
    return (
        pl.scan_csv(
            input_file,
            separator="\t",
            has_header=False,
            new_columns=["ctg", "monomer", "start", "stop", "identity"],
        )
        .with_columns(mon_id=pl.concat_str([pl.col("monomer", "identity")]))
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
    mid_mappings = {mid: chr(i) for i, mid in enumerate(mon_ids)}

    df_verkko = df_verkko.with_columns(pl.col("mon_id").replace(mid_mappings))
    df_hifiasm = df_hifiasm.with_columns(pl.col("mon_id").replace(mid_mappings))

    seq_verkko = "".join(df_verkko.get_column("mon_id").to_list())
    seq_hifiasm = "".join(df_hifiasm.get_column("mon_id").to_list())



if __name__ == "__main__":
    raise SystemExit(main())
