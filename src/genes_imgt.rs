use std::sync::LazyLock;
use std::collections::HashMap;

const GENE_REF: &[u8] = include_bytes!("imgt.202614-2.parsed.tsv");
const ALIASES: &[u8] = include_bytes!("gene_aliases.tsv");

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum Organism {
    Human,
    Mouse,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TcrChain {
    TRA,
    TRB,
    TRG,
    TRD,
    IGH,
    IGK,
    IGL,
}
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TcrGeneType {
    V,
    D,
    J,
}

#[derive(Debug)]
pub struct TcrData {
    pub id: (Organism, String),
    pub chain: TcrChain,
    pub gene_type: TcrGeneType,
    pub gene: String,
    pub sequence_nt: String,
    pub cdr3_sequence_nt: String,
    pub is_functional: bool,
}

impl Organism {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_lowercase().as_str() {
            "human" => Ok(Organism::Human),
            "mouse" => Ok(Organism::Mouse),
            other => Err(format!("Unknown organism: '{}'", other)),
        }
    }
}

impl TcrChain {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "TRA" => Ok(TcrChain::TRA),
            "TRB" => Ok(TcrChain::TRB),
            "TRG" => Ok(TcrChain::TRG),
            "TRD" => Ok(TcrChain::TRD),
            "IGH" => Ok(TcrChain::IGH),
            "IGK" => Ok(TcrChain::IGK),
            "IGL" => Ok(TcrChain::IGL),
            other => Err(format!("Unknown chain: '{}'", other)),
        }
    }
}

impl TcrGeneType {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "V" => Ok(TcrGeneType::V),
            "D" => Ok(TcrGeneType::D),
            "J" => Ok(TcrGeneType::J),
            other => Err(format!("Unknown gene type: '{}'", other)),
        }
    }
}

fn parse_gene_ref() -> impl Iterator<Item = Result<TcrData, String>> {
    let content = std::str::from_utf8(GENE_REF).expect("GENE_REF is not valid UTF-8");

    // Skip the header line, then parse each subsequent line
    // Expected columns : 0.species, 1.gene_type, 2.name, 3.gene, 4.chain, 5.sequence_nt, 6.cdr3_part_nt, 7.is_functional
    content
        .lines()
        .skip(1)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();

            if fields.len() < 8 {
                return Err(format!("Too few columns in line: '{}'", line));
            }

            let organism = Organism::parse(fields[0])?;
            let gene_type = TcrGeneType::parse(fields[1])?;
            let id_str = fields[2].to_string();
            let gene = fields[3].to_string();
            let chain = TcrChain::parse(fields[4])?;
            let sequence_nt = fields[5].to_string().to_ascii_lowercase();
            let cdr3_sequence_nt = fields[6].to_string().to_ascii_lowercase();
            let is_functional = fields[7] == "true";

            Ok(TcrData {
                id: (organism, id_str),
                chain,
                gene_type,
                gene,
                sequence_nt,
                cdr3_sequence_nt,
                is_functional,
            })
        })
}

pub static IMGT_REF: LazyLock<HashMap<Organism, HashMap<String, TcrData>>> =
    LazyLock::new(|| {
        let mut map: HashMap<Organism, HashMap<String, TcrData>> = HashMap::new();
        for entry in parse_gene_ref().map(|r| r.expect("Failed to parse GENE_REF line")) {
            let (organism, id) = entry.id.clone();
            map.entry(organism).or_default().insert(id, entry);
        }
        map
    });

// D-segment reference for TRB, keyed to the TRB locus only. The `chain == TRB` filter is
// load-bearing: the reference also contains TRD D-genes (TRDD1/2/3), and without it the "TRBD"
// index pools TRBD + TRDD, so TRB junctions get matched against TRD D-segments — a wrong
// annotation. `determine_reference_points` only computes D for TRB, so this index is TRB-only.
pub static IMGT_HUMAN_TRBD: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    IMGT_REF
        .get(&Organism::Human)
        .into_iter()
        .flat_map(|m| m.values())
        .filter(|t| t.gene_type == TcrGeneType::D && t.chain == TcrChain::TRB)
        .map(|t| t.sequence_nt.as_str())
        .collect()
});

pub static IMGT_MOUSE_TRBD: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    IMGT_REF
        .get(&Organism::Mouse)
        .into_iter()
        .flat_map(|m| m.values())
        .filter(|t| t.gene_type == TcrGeneType::D && t.chain == TcrChain::TRB)
        .map(|t| t.sequence_nt.as_str())
        .collect()
});


fn parse_aliases() -> impl Iterator<Item = (String, String)> {
    let content = std::str::from_utf8(ALIASES).expect("ALIASES is not valid UTF-8");

    content
        .lines()
        .skip(1)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != 2 {
                panic!("Expected exactly 2 columns in ALIASES line: '{}'", line);
            }
            (fields[0].to_string(), fields[1].to_string())
        })
}

pub static GENE_ALIASES: LazyLock<HashMap<String, String>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    for (alias, canonical) in parse_aliases() {
        map.insert(alias, canonical);
    }
    map
});