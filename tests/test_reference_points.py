"""Reference points from `_determine_reference_points` and the V(D)J features built on them."""
import polars as pl
import pytest

from tcr_io.expressions import _determine_reference_points

# IMGT junction parts from tcrio's reference (lowercase)
TRBV20_1 = "tgcagtgctagaga"
TRBV7_9 = "tgtgccagcagcttagc"
TRBJ1_1 = "tgaacactgaagctttcttt"
TRBJ2_1 = "ctcctacaatgagcagttcttc"


def refpoints(junction: str, v_call: str, j_call: str) -> dict | None:
    """Reference points of one junction as a dict, or None when no points were found."""
    out = (
        pl.DataFrame({"junction": [junction], "v_call": [v_call], "j_call": [j_call]})
        .select(_determine_reference_points(pl.col("junction"), pl.col("v_call"), pl.col("j_call")).alias("rp"))
        .to_series()
        .to_list()[0]
    )
    return None if out is None or out["v_end_trimmed"] is None else out


def test_exact_v_and_j_ends():
    rp = refpoints(TRBV20_1 + "ccc" + TRBJ1_1, "TRBV20-1*01", "TRBJ1-1*01")
    assert (rp["v_end_trimmed"], rp["v_end"]) == (14, 14)
    assert (rp["j_begin_trimmed"], rp["j_begin"], rp["cdr3_end"]) == (17, 17, 37)


def test_v_end_crosses_a_mismatch_followed_by_matches():
    # TRBV7-9 with t->g at position 14: the three matching bases after it keep the alignment going
    rp = refpoints("tgtgccagcagctgagc" + "tat" + TRBJ2_1, "TRBV7-9*01", "TRBJ2-1*01")
    assert rp["v_end_trimmed"] == 17
    assert rp["j_begin_trimmed"] == 20


def test_p_nucleotides_extend_v_and_j():
    # "tc" continues V as the reverse complement of its end; "ca" precedes J the same way
    rp = refpoints(TRBV20_1 + "tc" + "gg" + "ca" + TRBJ1_1, "TRBV20-1*01", "TRBJ1-1*01")
    assert (rp["v_end_trimmed"], rp["v_end"]) == (16, 14)
    assert (rp["j_begin_trimmed"], rp["j_begin"]) == (18, 20)


def test_unknown_gene_gives_no_points():
    assert refpoints(TRBV20_1 + "ccc" + TRBJ1_1, "TRBV99*01", "TRBJ1-1*01") is None


def test_overlapping_v_and_j_meet_at_the_midpoint():
    # J's first base is missing and V's P segment claims the next one: V ends at 18, J starts at 17
    rp = refpoints(TRBV7_9 + TRBJ1_1[1:], "TRBV7-9*01", "TRBJ1-1*01")
    assert rp["v_end_trimmed"] == rp["j_begin_trimmed"] == 17


@pytest.mark.parametrize(
    ("n_region", "d_call", "d_trimmed", "d_germline"),
    [
        # TRBD2 covering the G/A at position 13: G is *01, A is *02
        ("c" + "actagcgggggg" + "tc", "TRBD2*01", (15, 27), (12, 28)),
        ("c" + "actagcgggagg" + "tc", "TRBD2*02", (15, 27), (12, 28)),
        # TRBD2 trimmed before the SNP fits both alleles
        ("c" + "actagcggg" + "tc", "TRBD2*01,TRBD2*02", (15, 24), (12, 28)),
        # "gggac" starts both D genes; positions come from the TRBD1 placement (fewest deletions)
        ("ct" + "gggac" + "cc", "TRBD1*01,TRBD2*01,TRBD2*02", (16, 21), (16, 28)),
        # full TRBD1 with two P-nucleotides at each end: the germline D lies inside the match
        ("a" + "cc" + "gggacagggggc" + "gc" + "tt", "TRBD1*01", (15, 31), (17, 29)),
    ],
)
def test_d_assignment(n_region, d_call, d_trimmed, d_germline):
    rp = refpoints(TRBV20_1 + n_region + TRBJ1_1, "TRBV20-1*01", "TRBJ1-1*01")
    assert rp["v_end_trimmed"] == 14
    assert rp["d_call"] == d_call
    assert (rp["d_begin_trimmed"], rp["d_end_trimmed"]) == d_trimmed
    assert (rp["d_begin"], rp["d_end"]) == d_germline


@pytest.mark.parametrize(
    "n_region",
    [
        "ttgggatt",  # longest D match is 4 nt ("ggga"), below the 5-nt minimum
        "agctagtc",  # matches only the reverse complement of TRBD2: an inverted D is not indexed
    ],
)
def test_no_d(n_region):
    rp = refpoints(TRBV7_9 + n_region + TRBJ1_1, "TRBV7-9*01", "TRBJ1-1*01")
    assert (rp["v_end_trimmed"], rp["j_begin_trimmed"]) == (17, 25)
    assert [rp[k] for k in ("d_begin_trimmed", "d_end_trimmed", "d_begin", "d_end", "d_call")] == [None] * 5



def test_d_may_not_have_more_p_nucleotides_than_germline_bases():
    # "tcccg" is 4 P-nucleotides before a single TRBD base: not a D
    rp = refpoints(TRBV20_1 + "a" + "tcccg" + "tt" + TRBJ1_1, "TRBV20-1*01", "TRBJ1-1*01")
    assert rp["d_call"] is None and rp["d_begin_trimmed"] is None


def test_d_with_as_many_p_nucleotides_as_germline_bases():
    # "cccggg": 3 P-nucleotides before the first 3 bases of a TRB D gene
    rp = refpoints(TRBV20_1 + "a" + "cccggg" + "tt" + TRBJ1_1, "TRBV20-1*01", "TRBJ1-1*01")
    assert rp["d_call"] == "TRBD1*01,TRBD2*01,TRBD2*02"
    assert (rp["d_begin_trimmed"], rp["d_end_trimmed"]) == (15, 21)
    assert (rp["d_begin"], rp["d_end"]) == (18, 30)

def test_equally_good_d_matches_pool_their_alleles():
    # a *01-specific and a *02-specific 10-nt TRBD2 match with the same deletions: the call keeps
    # both alleles and the position of the first
    rp = refpoints(TRBV7_9 + "t" + "actagcgggg" + "t" + "actagcggga" + "t" + TRBJ1_1, "TRBV7-9*01", "TRBJ1-1*01")
    assert rp["d_call"] == "TRBD2*01,TRBD2*02"
    assert (rp["d_begin_trimmed"], rp["d_end_trimmed"]) == (18, 28)


def test_no_d_search_outside_trb():
    # TRAV1-2/TRAJ33 junction whose N region holds a TRBD2 core
    rp = refpoints("tgtgctgtgagaga" + "ggggactagcgg" + "tggatagcaactatcagttaatctgg", "TRAV1-2*01", "TRAJ33*01")
    assert (rp["v_end_trimmed"], rp["j_begin_trimmed"]) == (14, 26)
    assert rp["d_call"] is None and rp["d_begin_trimmed"] is None


def test_overlapping_v_and_j_leave_no_room_for_d():
    rp = refpoints(TRBV7_9 + TRBJ1_1[1:], "TRBV7-9*01", "TRBJ1-1*01")
    assert rp["v_end_trimmed"] == rp["j_begin_trimmed"] == 17
    assert rp["d_call"] is None and rp["d_begin"] is None


def test_germline_points_can_lie_outside_the_junction():
    # 12-nt junction: the germline J start falls 8 nt before the junction, the V end after it
    rp = refpoints("tgtgccagcttt", "TRBV7-9*01", "TRBJ1-1*01")
    assert (rp["v_end_trimmed"], rp["j_begin_trimmed"]) == (8, 8)
    assert (rp["v_end"], rp["j_begin"], rp["cdr3_end"]) == (17, -8, 12)


def test_output_types():
    schema = (
        pl.DataFrame({"junction": [TRBV20_1 + "ccc" + TRBJ1_1], "v_call": ["TRBV20-1*01"], "j_call": ["TRBJ1-1*01"]})
        .select(_determine_reference_points(pl.col("junction"), pl.col("v_call"), pl.col("j_call")).alias("rp"))
        .unnest("rp")
        .schema
    )
    assert dict(schema) == {
        "v_end_trimmed": pl.UInt32, "d_begin_trimmed": pl.UInt32, "d_end_trimmed": pl.UInt32,
        "j_begin_trimmed": pl.UInt32, "v_end": pl.Int32, "d_begin": pl.Int32, "d_end": pl.Int32,
        "j_begin": pl.Int32, "cdr3_end": pl.UInt32, "d_call": pl.String,
    }


from tcr_io.operations.vdj_statistics import determine_trim_features  # noqa: E402


def test_trim_features_split_deletions_p_and_insertions():
    df = pl.DataFrame({
        "junction": [
            TRBV20_1 + "tc" + "gg" + "ca" + TRBJ1_1,                         # 2 P after V, 2 P before J, no D
            TRBV20_1 + "a" + "cc" + "gggacagggggc" + "gc" + "tt" + TRBJ1_1,  # full TRBD1, 2 P at each end
            TRBV20_1 + "c" + "actagcgggggg" + "tc" + TRBJ1_1,               # TRBD2*01, 3 + 1 D bases deleted
            TRBV20_1 + "ccc" + TRBJ1_1,                                      # unknown V below: no points
        ],
        "v_call": ["TRBV20-1*01"] * 3 + ["TRBV99*01"],
        "j_call": ["TRBJ1-1*01"] * 4,
    })
    out = determine_trim_features(df.lazy()).collect().row(0, named=True)
    features = [
        "n_v_trim", "n_v_p", "n_vd_trim", "n_d5_p", "n_vj_trim", "n_d3_p", "n_j_trim", "n_j_p",
        "n_vd_ins", "n_dj_ins", "n_vj_ins", "n_trims", "n_p", "n_ins",
    ]
    assert {k: out[f"{k}_sum"] for k in features} == {
        "n_v_trim": 0, "n_v_p": 2, "n_vd_trim": 3, "n_d5_p": 2, "n_vj_trim": 1, "n_d3_p": 2,
        "n_j_trim": 0, "n_j_p": 2, "n_vd_ins": 2, "n_dj_ins": 4, "n_vj_ins": 2,
        "n_trims": 4, "n_p": 8, "n_ins": 8,
    }
    # the unknown-V row is left out, not counted as zero insertions
    assert out["n_ins_mean"] == pytest.approx(8 / 3)
    # D features describe only the two clones with a D
    assert out["n_vd_trim_mean"] == pytest.approx(1.5)
