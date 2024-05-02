import argparse
import polars as pl

RGX_HAP1 = r"(h1|haplotype1|pat)"
RGX_HAP2 = r"(h2|haplotype2|mat)"


def main():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("-i", "--infile", help="Input centromere contigs by chromosome")

    args = ap.parse_args()

    df = pl.concat(
        pl.scan_csv(
            r["path"],
            separator="\t",
            new_columns=["ctg", "length", "offset", "linebases", "linewidth"],
        )
        .with_columns(
            chr=pl.lit(r["chromosome"]),
            ort=pl.lit(r["orientation"]),
            grp=pl.lit(r["group"]),
            ctg_coord=pl.col("ctg").str.splitn(by=":", n=2),
        )
        .unnest("ctg_coord")
        .rename({"field_0": "ctg_name", "field_1": "ctg_coord"})
        .drop("ctg", "ctg_coord")
        .collect()
        # Open fasta index
        for r in pl.read_csv(args.infile, separator="\t", has_header=True)
        .with_columns(pl.col("path") + ".fai")
        .iter_rows(named=True)
    )
    df = (
        df.with_columns(
            # Expects sm_chr_hap-ctg
            # hap-ctg must not contains underscores.
            sm=pl.Series([ctg_name.rsplit("_", 2)[0] for ctg_name in df["ctg_name"]]),
            hap1=pl.col("ctg_name").str.extract(RGX_HAP1),
            hap2=pl.col("ctg_name").str.extract(RGX_HAP2),
        )
        .with_columns(
            hap=pl.when((pl.col("hap1").is_null()) & (pl.col("hap2").is_null()))
            .then(None)
            .when(~pl.col("hap1").is_null())
            .then(pl.lit("h1"))
            .otherwise(pl.lit("h2"))
        )
        .select("ctg_name", "sm", "hap", "grp", "ort", "chr")
    )
    # Group by sample, haplotype and chromosome.
    # Filter groups with only one contig.
    # Filter groups with only one grouping.
    for grp_name, df_grp in df.group_by(["sm", "hap", "chr"]):
        if df_grp.shape[0] < 2:
            continue
        if len(df_grp["grp"].unique()) == 1:
            continue
        print(grp_name)
        print(df_grp)
    breakpoint()


if __name__ == "__main__":
    raise SystemExit(main())
