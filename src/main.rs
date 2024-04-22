use std::collections::HashMap;

use eyre::Result;
use polars::{
    lazy::dsl::{col, concat_str},
    prelude::*,
};
use bio::alignment::pairwise::Scoring;
use bio::alignment::poa::*;


const SCHEMA_STRDEC_DATA: [(&str, DataType); 5] = [
    ("ctg", DataType::String),
    ("monomer", DataType::String),
    ("start", DataType::Int64),
    ("stop", DataType::Int64),
    ("identity", DataType::Float64),
];

fn read_format_strdec_data(input_file: &str) -> PolarsResult<DataFrame> {
    let mut schema: Schema = Schema::new();

    for (name, dtype) in SCHEMA_STRDEC_DATA.into_iter() {
        schema.with_column(name.to_string().into(), dtype);
    }

    let df = CsvReader::from_path(input_file)?
        .has_header(false)
        .with_separator(b'\t')
        .with_schema(Some(schema.into()))
        .finish()?;

    df.lazy()
        .with_column(concat_str([col("monomer"), col("identity")], "", true).alias("mon_id"))
        .collect()
}

fn convert_mon_ids_to_seq(df: &DataFrame, mid_mappings: &HashMap<String, char>) -> Result<String> {
    Ok(df.column("mon_id")?
        .str()?
        .into_iter()
        .flat_map(|val| {
            if let Some(val) = val.and_then(|v| mid_mappings.get(v)) {
                Some(val)
            } else {
                None
            }
        })
        .collect::<String>())
}
fn main() -> Result<()> {
    let df_hifiasm = read_format_strdec_data("data/hifiasm.tsv")?;
    let df_verkko = read_format_strdec_data("data/verkko.tsv")?;

    let mid_mappings: HashMap<String, char> = df_hifiasm
        .column("mon_id")?
        .iter()
        .chain(df_verkko.column("mon_id")?.iter())
        .enumerate()
        .flat_map(|(i, mid)| {
            mid.get_str().map(|mid| {
                (
                    mid.to_owned(),
                    char::from_u32(i as u32).expect("Number of mappings exceeds u32 limit."),
                )
            })
        })
        .collect();
    
    let seq_hifiasm = convert_mon_ids_to_seq(&df_hifiasm, &mid_mappings)?;
    let seq_verkko = convert_mon_ids_to_seq(&df_verkko, &mid_mappings)?;

    // char looks identical. sanity check.
    assert_ne!(mid_mappings.get("D'55.14").unwrap(), mid_mappings.get("I'58.29").unwrap());

    let scoring = Scoring::new(-1, 0, |a: u8, b: u8| if a == b { 1i32 } else { -1i32 });
    let mut aligner = Aligner::new(scoring, seq_hifiasm.as_bytes());

    aligner.global(seq_verkko.as_bytes()).add_to_graph();

    let consensus = aligner.consensus();
    println!("{:?}", consensus);

    Ok(())
}
