import os
import argparse
import polars as pl


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
DEF_CENS_INFO_COLS = [
    "ctg",
    "seq_path",
    "sm",
    "hap",
    "group",
    "orientation",
    "chromosome",
    "ctg_length",
]
DEF_OUTCFG_COLS = ["strdec_path", "misasm_path", "group", "orientation", "ctg_length"]


def main():
    ap = argparse.ArgumentParser(
        description="Generate config file for MSA of stringdecomposer output."
    )
    ap.add_argument(
        "-i",
        "--infile",
        required=True,
        help="Stringdecomposer output of a multiple contigs.",
    )
    ap.add_argument(
        "-c",
        "--contig_mdata",
        required=True,
        help=f"Contig metadata. Expects columns: {DEF_CENS_INFO_COLS}",
    )
    ap.add_argument(
        "-m",
        "--misasm_data",
        required=True,
        help="Misassemblies bed files by chromosome and group.",
    )
    ap.add_argument(
        "-o",
        "--output_cfg",
        default="msa_strdec_cfg.tsv",
        help=f"Output config file with columns: {DEF_OUTCFG_COLS}.",
    )
    ap.add_argument(
        "--output_str_dec_dir",
        default=None,
        help="Output dir to split stringdecomposer output by contig. Defaults to same dir as --output_cfg.",
    )

    args = ap.parse_args()

    # "output/stringdecomposer/K1463_2187_h1_chr1.tsv"
    df_strdec = pl.read_csv(
        args.infile, separator="\t", has_header=False, new_columns=DEF_STRDEC_COLS
    )
    # "output/groups/K1463_2187_h1_chr1.tsv"
    df_ctg_data = pl.read_csv(
        args.contig_mdata,
        separator="\t",
        has_header=False,
        new_columns=DEF_CENS_INFO_COLS,
    )
    # "cen_ctgs_misassemblies.tsv"
    df_misasm_data = pl.read_csv(
        args.misasm_data, separator="\t", has_header=True
    ).rename({"path": "misasm_path"})

    if args.output_str_dec_dir:
        output_dir = args.output_str_dec_dir
    else:
        output_dir = os.path.dirname(args.output_cfg)

    os.makedirs(output_dir, exist_ok=True)

    # Split stringdecomposer file by contig.
    # Save filenames to store in config.
    cen_strdec_files = {"ctg": [], "strdec_path": []}
    for cen_name, df_cen in df_strdec.group_by(["cen"]):
        cen_name = cen_name[0]
        cen_strdec_path = os.path.join(output_dir, f"{cen_name}.tsv")
        cen_strdec_files["ctg"].append(cen_name)
        cen_strdec_files["strdec_path"].append(cen_strdec_path)
        df_cen.write_csv(cen_strdec_path, separator="\t", include_header=False)

    df_ctg_data = (
        df_ctg_data.join(df_misasm_data, on=["chromosome", "group"], how="left")
        .join(pl.from_dict(cen_strdec_files), on=["ctg"], how="left")
        .with_columns(pl.col("orientation").replace({"forward": "+", "reverse": "-"}))
        .select("strdec_path", "misasm_path", "group", "orientation", "ctg_length")
    )
    df_ctg_data.write_csv(args.output_cfg, separator="\t", include_header=True)


if __name__ == "__main__":
    raise SystemExit(main())
