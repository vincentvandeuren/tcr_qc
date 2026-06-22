from typing import TYPE_CHECKING, List, Literal
import polars as pl
from .base import BaseOperation, OperationResults
from tqdm import tqdm

if TYPE_CHECKING:
    from tcr_io.dataset import TcrDataset

class OverlapAnalyzer(BaseOperation):
    name = "overlap_analyzer"
    version = "0.1"
    description = "Analyzes the overlap between repertoires, calculating the number of shared clonotypes and the Spearman correlation between their frequencies."

    def __init__(
            self,
            top_n: int = 250,
            junction_col : Literal["junction", "junction_aa"] = "junction",
            threshold_sus_overlap :int = 40, # different patient, above this threshold, the overlap is suspicious 
            threshold_sus_overlap_absence: int = 20 # same patient, below this threshold, the absence of overlap is suspicious
            ):
        self.top_n = top_n
        self.junction_col = junction_col
        self.threshold_sus_overlap = threshold_sus_overlap
        self.threshold_sus_overlap_absence = threshold_sus_overlap_absence

    def _run(self, ds) -> OperationResults:

        head_dir = ds.db_dir / "temp/heads"
        head_dir.mkdir(exist_ok=True, parents=True)

        for repertoire_id, df in ds.iter_repertoires(progress_bar=True, progress_desc="Collecting heads"):
            repertoire_id_safe = repertoire_id.replace("/", "_")
            df.head(self.top_n).with_columns(
                v_gene = pl.col("v_call").str.extract(r"(.*)\*\d{2}"),
                j_gene = pl.col("j_call").str.extract(r"(.*)\*\d{2}"),
            ).sink_parquet(head_dir / f"{repertoire_id_safe}.parquet")

        heads_tab_dir = ds.db_dir / "temp/heads_tab"
        heads_tab_dir.mkdir(exist_ok=True, parents=True)

        pl.scan_parquet(head_dir).select(["v_gene", "j_gene", self.junction_col, "repertoire_id"]).sink_parquet(
            pl.PartitionBy(heads_tab_dir, key=["v_gene", "j_gene"])
        )
    
        total_dirs = sum(1 for _ in heads_tab_dir.glob("*/*"))
        progress_bar = tqdm(total=total_dirs, desc="Finding shared clones")
        shared_clns_list = []
        for v in heads_tab_dir.iterdir():
            for j in v.iterdir():
                shared_clns = pl.scan_parquet(j).group_by(["v_gene", "j_gene", self.junction_col]).agg(
                    pl.len().alias("count")
                ).filter(pl.col("count") > 1).collect(engine="streaming")
                shared_clns_list.append(shared_clns)
                progress_bar.update(1)

        shared_clns_list = pl.concat(shared_clns_list).sort("count", descending=True).with_row_index("clone_id")
        overlap = pl.scan_parquet(
            head_dir
        ).join(
            shared_clns_list.lazy(),
            on= ["v_gene", "j_gene", self.junction_col],
            how="inner"
        )
        overlap = overlap.join(
            overlap, on="clone_id", how="inner"
        ).filter(
            pl.col("repertoire_id") < pl.col("repertoire_id_right")
        ).group_by(["repertoire_id", "repertoire_id_right"]).agg(
            pl.len().alias("n_overlap"),
            pl.corr(pl.col("duplicate_count"), pl.col("duplicate_count_right"), method="pearson").fill_nan(0).alias("pearson_corr"),
            pl.corr(pl.col("duplicate_count"), pl.col("duplicate_count_right"), method="spearman").fill_nan(0).alias("spearman_corr"),
        ).collect(engine="streaming")

        o = ds.repertoire_meta.select(["repertoire_id", "patient_id", "n_filtered_clonotypes"]).with_columns(join=pl.lit(True))

        overlap_table = o.join(o, on="join", how="full").filter(pl.col("repertoire_id") < pl.col("repertoire_id_right")).drop(["join", "join_right"]).join(
            overlap, on=["repertoire_id", "repertoire_id_right"], how="left"
        ).with_columns(
            pl.col("n_overlap").fill_null(0),
            pl.col("pearson_corr").fill_null(0),
            pl.col("spearman_corr").fill_null(0),
            pl.col("patient_id").eq(pl.col("patient_id_right")).alias("same_patient")
        ).sort(
            "n_overlap", descending=True
        )

        overlap_table = overlap_table.with_columns(
            flag_sus_overlap = (~pl.col("same_patient")).and_(pl.col("n_overlap")>self.threshold_sus_overlap),
            flag_sus_overlap_absence = pl.col("same_patient").and_(pl.col("n_overlap")<self.threshold_sus_overlap_absence)
        )

        overlap_overview = overlap_table.select(
            pl.len().alias("total_pairs"),
            pl.col("flag_sus_overlap").sum().alias("sus_overlap"),
            pl.col("flag_sus_overlap_absence").sum().alias("sus_overlap_absence")
        ).with_columns(
            n_repertoires = ds.repertoire_meta.height
        )

        return OperationResults(
            outputs={
                "qc/overlap_table.parquet": overlap_table,
                "qc/overlap_overview.parquet": overlap_overview
            }
        )


def plot_overlap(ds, x="hla_dist", y="n_overlap"):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    overlap_df = ds.get_operation_result("overlap_analyzer")
    hla_dist_df = ds.get_operation_result("repertoire_hla_inference", "dist")
    overlap_df = overlap_df.join(hla_dist_df, on=["repertoire_id", "repertoire_id_right"], how="left")

    q5, q95 = np.quantile(
        overlap_df["hla_dist"].drop_nulls().to_numpy(),
        [0.05, 0.95]
    )

    fig, axs = plt.subplots(figsize=(8,6), nrows=2, gridspec_kw={"height_ratios":[4,1]}, sharex="col")

    sns.scatterplot(
        data = overlap_df,
        x=x,
        y=y,
        hue="same_patient",
        marker="x",
        size="spearman_corr",
        size_norm=(0,1),
        ax=axs[0]
    )

    sns.kdeplot(
        data = overlap_df,
        x="hla_dist",
        ax=axs[1]
    )

    axs[0].axhline(50, color=(".3", .3), linestyle="--")

    for ax in axs:
        ax.axvline(q5, color=(".3",.3), linestyle="--")
        ax.axvline(q95, color=(".3",.3), linestyle="--")

    return fig, axs


def plot_interactive_overlap(ds, x="hla_dist", y="n_overlap"):
    import plotly.express as px
    overlap_df = ds.get_operation_result("overlap_analyzer")
    hla_dist_df = ds.get_operation_result("repertoire_hla_inference", "dist")
    overlap_df = overlap_df.join(hla_dist_df, on=["repertoire_id", "repertoire_id_right"], how="left")


    fig = px.scatter(
        overlap_df.filter(
            pl.col("n_overlap")>1
        ),
        x=x,
        y=y,
        color="same_patient",
        hover_data=["repertoire_id", "repertoire_id_right", "patient_id", "patient_id_right"],
    )
    return fig


from typing import List, Literal

def plot_sus_overlap_graph(ds, patient_ids: List[str], figsize=(8, 6),
                           layout: Literal["kamada_kawai", "spring", "circular"] = "kamada_kawai",
                           k=0.5, show_non_sus=True, show_external_sus=True,
                           minimize_non_sus_overlap=True):
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import numpy as np
    import matplotlib
    import colorcet as cc
    import seaborn as sns

    overlap_df = ds.get_operation_result("overlap_analyzer")
    hla_dist_df = ds.get_operation_result("repertoire_hla_inference", "dist")
    overlap_df = overlap_df.join(hla_dist_df, on=["repertoire_id", "repertoire_id_right"], how="left")

    q5, q95 = np.quantile(
        overlap_df["hla_dist"].drop_nulls().to_numpy(),
        [0.05, 0.95]
    )

    # Filter to relevant overlaps
    relevant_df = overlap_df.filter(
        pl.col("patient_id").is_in(patient_ids) | pl.col("patient_id_right").is_in(patient_ids)
    )

    both_selected = pl.col("patient_id").is_in(patient_ids) & pl.col("patient_id_right").is_in(patient_ids)
    is_sus = pl.col("flag_sus_overlap") | pl.col("flag_sus_overlap_absence")

    conditions = [both_selected & is_sus]
    if show_non_sus:
        conditions.append(both_selected & ~is_sus)
    if show_external_sus:
        conditions.append(~both_selected & is_sus)

    combined = conditions[0]
    for c in conditions[1:]:
        combined = combined | c

    plot_df = relevant_df.filter(combined).sort("patient_id", "patient_id_right")

    # Build graph
    G = nx.Graph()
    node_patient = {}
    patient_ids_set = set(patient_ids)

    for row in plot_df.iter_rows(named=True):
        rid, rid_r = row["repertoire_id"], row["repertoire_id_right"]
        pid, pid_r = row["patient_id"], row["patient_id_right"]
        node_patient[rid] = pid
        node_patient[rid_r] = pid_r

        if not (row["flag_sus_overlap"] or row["flag_sus_overlap_absence"]):
            etype = "non_sus"
        elif row["flag_sus_overlap_absence"]:
            etype = "sus_absence"
        else:
            etype = "sus_overlap"

        G.add_edge(rid, rid_r,
                    weight=row["n_overlap"],
                    hla_dist=row["hla_dist"],
                    edge_type=etype)

    for node, patient in node_patient.items():
        G.add_node(node, patient_id=patient, is_external=patient not in patient_ids_set)

    # Styling
    patients = sorted(set(node_patient.values()))
    palette = {p: c for p, c in zip(patients, cc.glasbey_dark[:len(patients)])}

    nodes_internal = [n for n in G.nodes() if not G.nodes[n]["is_external"]]
    nodes_external = [n for n in G.nodes() if G.nodes[n]["is_external"]]

    edges_by_type = {
        "sus_overlap": [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "sus_overlap"],
        "sus_absence": [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "sus_absence"],
        "non_sus":     [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "non_sus"],
    }

    all_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(max(all_weights, default=1), 1)

    def get_widths(edge_list):
        return [1 + 5 * G[u][v]["weight"] / max_w for u, v in edge_list]

    def get_hla_vals(edge_list):
        mid = (q5 + q95) / 2
        return [G[u][v]["hla_dist"] if G[u][v]["hla_dist"] is not None else mid
                for u, v in edge_list]

    # Layout
    norm = mcolors.Normalize(vmin=q5, vmax=q95)
    cmap = matplotlib.colormaps["cividis"]

    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=k)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        raise ValueError(f"Invalid layout: {layout}")

    fig, ax = plt.subplots(figsize=figsize)

    # Non-sus edges
    if edges_by_type["non_sus"]:
        non_sus_nonzero = [(u, v) for u, v in edges_by_type["non_sus"] if G[u][v]["weight"] > 0]
        if minimize_non_sus_overlap:
            nx.draw_networkx_edges(G, pos, edgelist=non_sus_nonzero,
                                   width=0.8, edge_color="lightgray",
                                   style="dotted", alpha=0.2, ax=ax)
        else:
            if non_sus_nonzero:
                nx.draw_networkx_edges(G, pos, edgelist=non_sus_nonzero,
                                       width=get_widths(non_sus_nonzero),
                                       edge_color=get_hla_vals(non_sus_nonzero),
                                       edge_cmap=cmap, edge_vmin=q5, edge_vmax=q95,
                                       style="dotted", alpha=0.8, ax=ax)

    # Sus overlap (solid, HLA colored)
    if edges_by_type["sus_overlap"]:
        nx.draw_networkx_edges(G, pos, edgelist=edges_by_type["sus_overlap"],
                               width=get_widths(edges_by_type["sus_overlap"]),
                               edge_color=get_hla_vals(edges_by_type["sus_overlap"]),
                               edge_cmap=cmap, edge_vmin=q5, edge_vmax=q95,
                               alpha=0.8, ax=ax)

    # Sus absence (dashed, HLA colored)
    if edges_by_type["sus_absence"]:
        nx.draw_networkx_edges(G, pos, edgelist=edges_by_type["sus_absence"],
                               width=get_widths(edges_by_type["sus_absence"]),
                               edge_color=get_hla_vals(edges_by_type["sus_absence"]),
                               edge_cmap=cmap, edge_vmin=q5, edge_vmax=q95,
                               style="dashed", alpha=0.8, ax=ax)

    # Internal nodes (large circles)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_internal,
                           node_color=[palette[node_patient[n]] for n in nodes_internal],
                           node_size=800, edgecolors="black", linewidths=1.5, ax=ax)

    # External nodes (smaller squares)
    if nodes_external:
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_external,
                               node_color=[palette[node_patient[n]] for n in nodes_external],
                               node_size=400, edgecolors="gray", linewidths=1.0,
                               node_shape="s", ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=6, font_weight="bold", ax=ax)

    # Edge labels
    if minimize_non_sus_overlap:
        labeled_edges = edges_by_type["sus_overlap"] + edges_by_type["sus_absence"]
    else:
        labeled_edges = list(G.edges())
    edge_labels = {(u, v): G[u][v]["weight"] for u, v in labeled_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.2, pad=0.02)
    cbar.set_label("HLA distance")

    # Legend
    handles = [mpatches.Patch(color=palette[p], label=f"Patient {p}") for p in patients]
    handles.append(plt.Line2D([], [], color="gray", linewidth=2, label="Sus overlap (solid)"))
    handles.append(plt.Line2D([], [], color="gray", linewidth=2, linestyle="dashed",
                              label="Sus absence (dashed)"))
    if show_non_sus:
        handles.append(plt.Line2D([], [], color="lightgray", linewidth=1, linestyle="dotted",
                                  label="Non-sus overlap"))
    if nodes_external:
        handles.append(plt.Line2D([], [], marker="s", color="w", markerfacecolor="gray",
                                  markersize=8, label="External sample"))
    ax.legend(handles=handles, loc="upper left")

    ax.set_title("Suspicious Sample Overlaps")
    ax.axis("off")
    plt.tight_layout()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    return fig, ax