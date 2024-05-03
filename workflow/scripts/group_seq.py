import os
import argparse
import polars as pl

RGX_HAP1 = r"(h1|haplotype1|pat)"
RGX_HAP2 = r"(h2|haplotype2|mat)"


def read_cens_info(infile: str) -> pl.DataFrame:
    df_cens_info = pl.concat(
        pl.scan_csv(
            r["path"] + ".fai",
            separator="\t",
            new_columns=["ctg", "length", "offset", "linebases", "linewidth"],
        )
        .with_columns(
            chr=pl.lit(r["chromosome"]),
            ort=pl.lit(r["orientation"]),
            grp=pl.lit(r["group"]),
            path=pl.lit(r["path"]),
            ctg_coord=pl.col("ctg").str.splitn(by=":", n=2),
        )
        .unnest("ctg_coord")
        .rename({"field_0": "ctg_name"})
        # Open fasta index
        for r in pl.read_csv(infile, separator="\t", has_header=True)
        .iter_rows(named=True)
    ).collect()
    return (
        df_cens_info.with_columns(
            # Expects sm_chr_hap-ctg
            # hap-ctg must not contains underscores.
            sm=pl.Series(
                [ctg_name.rsplit("_", 2)[0] for ctg_name in df_cens_info["ctg_name"]]
            ),
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
        .select("ctg_name", "ctg", "path", "sm", "hap", "grp", "ort", "chr")
    )


def read_misassembled_cens(infile: str) -> pl.Series:
    return (
        pl.concat(
            pl.scan_csv(
                r["path"],
                separator="\t",
                new_columns=["ctg", "start", "stop", "misassembly"],
            )
            .with_columns(
                ctg_coord=pl.col("ctg").str.splitn(by=":", n=2),
            )
            .unnest("ctg_coord")
            .rename({"field_0": "ctg_name", "field_1": "ctg_coord"})
            .drop("ctg", "ctg_coord")
            for r in pl.read_csv(infile, separator="\t", has_header=True).iter_rows(
                named=True
            )
        )
        .collect()
        .get_column("ctg_name")
        .unique()
    )

def read_assembly_contig_lengths(infile: str) -> pl.DataFrame:
    return (
        pl.concat(
            pl.scan_csv(
                r["path"],
                separator="\t",
                new_columns=["ctg_name", "ctg_length", "offset", "linebases", "linewidth"],
            )
            for r in pl.read_csv(infile, separator="\t", has_header=True).iter_rows(
                named=True
            )
        )
        .select("ctg_name", "ctg_length")
        .collect()
    )

def main():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("-i", "--infile", required=True, help="Centromere contigs fastas by chromosome, orientation, and group.")
    ap.add_argument("-m", "--misassemblies", required=True, help="Misassemblies bed files by chromosome and group.")
    ap.add_argument("-a", "--assembly_faidx", required=True, help="Assembly faidxs by group.")
    ap.add_argument("-o", "--output_dir", default="output", help="Output dir with contigs")

    args = ap.parse_args()

    srs_misassembled_cens = read_misassembled_cens(args.misassemblies)
    df_ctg_lengths = read_assembly_contig_lengths(args.assembly_faidx)
    df_cens_info = read_cens_info(args.infile).join(df_ctg_lengths, on="ctg_name", how="left")

    os.makedirs(args.output_dir)

    # Group by sample, haplotype and chromosome.
    for grp_name, df_grp in df_cens_info.group_by(["sm", "hap", "chr"]):
        # Filter groups with only one contig.
        if df_grp.shape[0] < 2:
            continue

        # Filter groups with only one grouping.
        if len(df_grp["grp"].unique()) == 1:
            continue

        # Filter group if no misassemblies.
        if df_grp.get_column("ctg_name").is_in(srs_misassembled_cens).all():
            continue
        
        df_grp.drop_in_place("ctg_name")
        df_grp.write_csv(
            os.path.join(
                args.output_dir, f"{'_'.join(grp_name)}.tsv"
            ),
            separator="\t",
            include_header=False
        )


if __name__ == "__main__":
    raise SystemExit(main())
