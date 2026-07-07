#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use crate::genes_imgt::{Organism, GENE_ALIASES, IMGT_REF};
use crate::reference_points::{determine_reference_points};
use crate::junction_trimmer::trim_junction_to_cdr3;

/// Declare the Struct dtype for ReferencePoints
fn reference_points_struct_output(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "reference_points".into(),
        DataType::Struct(vec![
            Field::new("v_end_trimmed".into(), DataType::UInt32),
            Field::new("d_begin_trimmed".into(), DataType::UInt32),
            Field::new("d_end_trimmed".into(), DataType::UInt32),
            Field::new("j_begin_trimmed".into(), DataType::UInt32),
            Field::new("v_end".into(), DataType::UInt32),
            Field::new("d_begin".into(), DataType::UInt32),
            Field::new("d_end".into(), DataType::UInt32),
            Field::new("j_begin".into(), DataType::UInt32),
            Field::new("cdr3_end".into(), DataType::UInt32),
        ]),
    ))
}

#[derive(Deserialize)]
struct OrganismKwargs {
    organism: String,
}

#[polars_expr(output_type_func=reference_points_struct_output)]
pub fn determine_reference_points_polars(
    inputs: &[Series],
    kwargs : OrganismKwargs,
) -> PolarsResult<Series> {
    let ca_junction: &StringChunked = inputs[0].str()?;
    let ca_v_call: &StringChunked = inputs[1].str()?;
    let ca_j_call: &StringChunked = inputs[2].str()?;

    let species= Organism::parse(&kwargs.organism).map_err(|e| PolarsError::ComputeError(e.into()))?;
    
    let len = ca_junction.len();

    let mut col_v_end_trimmed: Vec<Option<u32>> = Vec::with_capacity(len);
    let mut col_d_begin_trimmed: Vec<Option<u32>> = Vec::with_capacity(len);
    let mut col_d_end_trimmed: Vec<Option<u32>> = Vec::with_capacity(len);
    let mut col_j_begin_trimmed: Vec<Option<u32>> = Vec::with_capacity(len);
    let mut col_v_end: Vec<Option<u32>> = Vec::with_capacity(len);
    let mut col_d_begin: Vec<Option<u32>> = Vec::with_capacity(len);
    let mut col_d_end: Vec<Option<u32>> = Vec::with_capacity(len);
    let mut col_j_begin: Vec<Option<u32>> = Vec::with_capacity(len);
    let mut col_cdr3_end: Vec<Option<u32>> = Vec::with_capacity(len);

    for ((opt_junction, opt_v_call), opt_j_call) in ca_junction
        .into_iter()
        .zip(ca_v_call.into_iter())
        .zip(ca_j_call.into_iter())
    {
        match (opt_junction, opt_v_call, opt_j_call) {
            (Some(junction), Some(v_call), Some(j_call)) => {
                if let Some(rp) = determine_reference_points(species, junction, v_call, j_call) {
                    col_v_end_trimmed.push(Some(rp.v_end_trimmed as u32));
                    col_d_begin_trimmed.push(Some(rp.d_begin_trimmed as u32));
                    col_d_end_trimmed.push(Some(rp.d_end_trimmed as u32));
                    col_j_begin_trimmed.push(Some(rp.j_begin_trimmed as u32));
                    col_v_end.push(Some(rp.v_end as u32));
                    col_d_begin.push(Some(rp.d_begin as u32));
                    col_d_end.push(Some(rp.d_end as u32));
                    col_j_begin.push(Some(rp.j_begin as u32));
                    col_cdr3_end.push(Some(rp.cdr3_end as u32));
                } else {
                    // determine_reference_points returned None
                    col_v_end_trimmed.push(None);
                    col_d_begin_trimmed.push(None);
                    col_d_end_trimmed.push(None);
                    col_j_begin_trimmed.push(None);
                    col_v_end.push(None);
                    col_d_begin.push(None);
                    col_d_end.push(None);
                    col_j_begin.push(None);
                    col_cdr3_end.push(None);
                }
            }
            _ => {
                // Any input is null → propagate nulls
                col_v_end_trimmed.push(None);
                col_d_begin_trimmed.push(None);
                col_d_end_trimmed.push(None);
                col_j_begin_trimmed.push(None);
                col_v_end.push(None);
                col_d_begin.push(None);
                col_d_end.push(None);
                col_j_begin.push(None);
                col_cdr3_end.push(None);
            }
        }
    }

    let s_v_end_trimmed = Series::new("v_end_trimmed".into(), col_v_end_trimmed);
    let s_d_begin_trimmed = Series::new("d_begin_trimmed".into(), col_d_begin_trimmed);
    let s_d_end_trimmed = Series::new("d_end_trimmed".into(), col_d_end_trimmed);
    let s_j_begin_trimmed = Series::new("j_begin_trimmed".into(), col_j_begin_trimmed);
    let s_v_end = Series::new("v_end".into(), col_v_end);
    let s_d_begin = Series::new("d_begin".into(), col_d_begin);
    let s_d_end = Series::new("d_end".into(), col_d_end);
    let s_j_begin = Series::new("j_begin".into(), col_j_begin);
    let s_cdr3_end = Series::new("cdr3_end".into(), col_cdr3_end);

    let fields = vec![
        s_v_end_trimmed,
        s_d_begin_trimmed,
        s_d_end_trimmed,
        s_j_begin_trimmed,
        s_v_end,
        s_d_begin,
        s_d_end,
        s_j_begin,
        s_cdr3_end,
    ];

    StructChunked::from_series("reference_points".into(), len, fields.iter())
        .map(|ca| ca.into_series())
}

#[polars_expr(output_type=String)]
pub fn trim_junction_to_cdr3_polars(inputs: &[Series]) -> PolarsResult<Series> {
    let junction_nt: &StringChunked = inputs[0].str()?;
    let junction_aa: &StringChunked = inputs[1].str()?;

    let out: StringChunked = junction_nt
        .into_iter()
        .zip(junction_aa.into_iter())
        .map(|(nt, aa)| match (nt, aa) {
            (Some(nt), Some(aa)) => trim_junction_to_cdr3(nt, aa),
            _ => None,
        })
        .collect();

    Ok(out.into_series())
}


#[derive(Deserialize)]
struct GeneToImgtKwargs {
    split_on_character: String,
}

#[polars_expr(output_type=String)]
pub fn gene_to_imgt_canonical(inputs: &[Series], kwargs: GeneToImgtKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let split_on = &kwargs.split_on_character;

    let out = ca.apply_into_string_amortized(|v, buf| {
        let part = if split_on.is_empty() {
            v
        } else {
            v.split(split_on.as_str())
                .next()
                .map(str::trim)
                .unwrap_or(v)
        };

        if let Some((gene, allele)) = part.split_once('*') {
            let mapped = GENE_ALIASES
                .get(gene)
                .map(|s| s.as_str())
                .unwrap_or(gene);
            buf.push_str(mapped);
            buf.push('*');
            if allele == "00" {
                buf.push_str("01");
            } else {
                buf.push_str(allele);
            }
        } else {
            let mapped = GENE_ALIASES
                .get(part)
                .map(|s| s.as_str())
                .unwrap_or(part);
            buf.push_str(mapped);
            buf.push_str("*01");
        }
    });
    Ok(out.with_name(ca.name().clone()).into_series())
}

fn is_functional_struct_output(_input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "is_functional".into(),
        DataType::Struct(vec![
            Field::new("v_valid_imgt".into(), DataType::Boolean),
            Field::new("v_func_imgt".into(), DataType::Boolean),
            Field::new("j_valid_imgt".into(), DataType::Boolean),
            Field::new("j_func_imgt".into(), DataType::Boolean),
            Field::new("consensus".into(), DataType::Boolean),
        ]),
    ))
}

#[polars_expr(output_type_func=is_functional_struct_output)]
pub fn is_functional_tcr(inputs: &[Series], kwargs: OrganismKwargs) -> PolarsResult<Series> {
    let ca_v_call: &StringChunked = inputs[0].str()?;
    let ca_j_call: &StringChunked = inputs[1].str()?;
    let species= Organism::parse(&kwargs.organism).map_err(|e| PolarsError::ComputeError(e.into()))?;

    let ref_map = IMGT_REF.get(&species);
    let lookup = |call: &str| ref_map.and_then(|m| m.get(call));

    let len = ca_v_call.len();

    let (v_valid, v_func): (Vec<Option<bool>>, Vec<Option<bool>>) = ca_v_call.into_iter()
        .map(|opt| match opt {
            Some(v) => {
                let entry = lookup(v);
                (Some(entry.is_some()), Some(entry.map_or(false, |d| d.is_functional)))
            }
            None => (None, None),
        })
        .unzip();

    let (j_valid, j_func): (Vec<Option<bool>>, Vec<Option<bool>>) = ca_j_call.into_iter()
        .map(|opt| match opt {
            Some(j) => {
                let entry = lookup(j);
                (Some(entry.is_some()), Some(entry.map_or(false, |d| d.is_functional)))
            }
            None => (None, None),
        })
        .unzip();

    let functional: Vec<Option<bool>> = (0..len)
        .map(|i| {
            match (v_valid[i], j_valid[i], v_func[i], j_func[i]) {
                (Some(a), Some(b), Some(c), Some(d)) => Some(a && b && c && d),
                _ => Some(false),
            }
        })
        .collect();

    let fields = vec![
        Series::new("v_valid_imgt".into(), &v_valid),
        Series::new("v_func_imgt".into(), &v_func),
        Series::new("j_valid_imgt".into(), &j_valid),
        Series::new("j_func_imgt".into(), &j_func),
        Series::new("consensus".into(), &functional),
    ];

    StructChunked::from_series("is_functional".into(), len, fields.iter())
        .map(|ca| ca.into_series())
}
