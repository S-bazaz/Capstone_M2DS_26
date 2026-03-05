# -*- coding: utf-8 -*-
############
# Packages #
############
import sys
from pathlib import Path

import geopandas as gpd
import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

sys.path.append(str(Path(__file__).parents[2]))

#######################
# Internal Imports #
#######################
from dataclass.graph_data import GraphData

#########
# Utils #
#########


def build_networkx_graph(graph_data: GraphData) -> nx.Graph:
    """
    Construit un graphe NetworkX à partir d'un objet GraphData.
    """
    graph = nx.Graph()
    df_segments = graph_data.gdf_segments[['i', 'j', 'length_m']].copy()
    if df_segments is None or df_segments.empty:
        return graph

    for segment_idx, row in df_segments.iterrows():
        graph.add_edge(
            int(row["i"]),
            int(row["j"]),
            length_m=float(row["length_m"]),
            segment_idx=int(segment_idx),
        )

    if len(graph_data.gdf_nodes) > 0:
        nodes_coordinates = graph_data.nodes_coordinates
        nodes_pos_dict = {}
        for node_id in range(len(nodes_coordinates)):
            coord = nodes_coordinates[node_id]
            nodes_pos_dict[node_id] = tuple(coord)  # (x, y)
        nx.set_node_attributes(graph, nodes_pos_dict, "pos")

    return graph


def build_igraph_graph(graph_data: GraphData) -> ig.Graph:
    """
    Construit un graphe igraph à partir d'un objet GraphData.
    """
    df_segments = graph_data.gdf_segments[['i', 'j', 'length_m']].copy()
    if df_segments is None or df_segments.empty:
        return ig.Graph()

    i_nodes = df_segments["i"].astype(int).tolist()
    j_nodes = df_segments["j"].astype(int).tolist()
    graph = ig.Graph()
    if i_nodes or j_nodes:
        max_node_id = max(i_nodes + j_nodes)
        graph.add_vertices(max_node_id + 1)
    edges = list(zip(i_nodes, j_nodes))
    if edges:
        graph.add_edges(edges)
    if "length_m" in df_segments.columns:
        graph.es["length_m"] = df_segments["length_m"].astype(float).tolist()
    # Stocker l'index du segment dans df_segments
    graph.es["segment_idx"] = df_segments.index.astype(int).tolist()

    return graph


def build_networkx_segment_graph(graph_data: GraphData) -> nx.Graph:
    """
    Construit un graphe NetworkX où chaque nœud représente un segment.

    Dans ce graphe:
    - Les nœuds sont des segments identifiés par (i, j)
    - Les arêtes relient deux segments qui partagent un nœud
      (ex: segment (i,j) et (i,k) ou (i,j) et (j,k))
    - Chaque nœud a l'attribut length_m et pos (moyenne des positions)
    - Chaque arête a l'attribut abs_cosine provenant de df_angles
    """
    graph = nx.Graph()
    df_segments = graph_data.gdf_segments.copy()

    if df_segments is None or df_segments.empty:
        return graph

    if df_segments["length_m"].isna().any():
        nan_rows = df_segments[df_segments["length_m"].isna()]
        raise ValueError("length_m manquant sur au moins un segment")

    nodes_coordinates = graph_data.nodes_coordinates

    # Créer tous les nœuds (segments) en parcourant gdf_segments
    for segment_idx, row in df_segments.iterrows():
        i, j = int(row["i"]), int(row["j"])
        # Calculer la position moyenne pour les plots
        pos_i = nodes_coordinates[i]
        pos_j = nodes_coordinates[j]
        pos_mean = tuple((pos_i + pos_j) / 2.0)

        graph.add_node(
            (i, j),
            length_m=float(row["length_m"]),
            segment_idx=int(segment_idx),
            pos=pos_mean,
        )

    # Créer un set des segments pour vérification rapide
    segments_set = set(graph.nodes())

    if graph_data.df_angles is None:
        raise ValueError("df_angles est None alors que des segments existent.")

    for _, angle_row in graph_data.df_angles.iterrows():
        i = int(angle_row["i"])
        j = int(angle_row["j"])
        k = int(angle_row["k"])
        abs_cosine = float(angle_row["abs_cosine"])
        if pd.isna(abs_cosine):
            raise ValueError(
                f"abs_cosine NaN détecté pour l'angle ({i},{j},{k})"
            )

        # Les segments sont normalisés avec i < j
        seg1 = (min(i, j), max(i, j))
        seg2 = (min(j, k), max(j, k))
        if seg1 != seg2 and seg1 in segments_set and seg2 in segments_set:
            graph.add_edge(seg1, seg2, abs_cosine=abs_cosine)
           
    return graph


def test_igraph_build() -> None:
    """
    Test de base pour la construction Igraph.
    """
    # Créer les géométries LineString pour les segments
    geometries_segments = [
        LineString([(0.0, 0.0), (2.0, 1.0)]),
        LineString([(2.0, 1.0), (2.0, 2.0)]),
    ]
    gdf_segments = gpd.GeoDataFrame(
        {
            "i": [0, 2],
            "j": [2, 3],
            "length_m": [7.0, 4.0],
            "capacity": [120, 60],
        },
        geometry=geometries_segments,
    )

    # Créer les géométries Point pour les nœuds
    geometries_nodes = [
        Point(0.0, 0.0),
        Point(2.0, 0.0),
        Point(2.0, 1.0),
        Point(2.0, 2.0),
    ]
    gdf_nodes = gpd.GeoDataFrame(
        {"node_id": [0, 1, 2, 3]},
        geometry=geometries_nodes,
    )

    # nodes_coordinates est de shape (max_node_id+1, 2)
    nodes_coordinates = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )

    graph_input = GraphData(
        gdf_segments=gdf_segments,
        gdf_nodes=gdf_nodes,
        nodes_coordinates=nodes_coordinates,
        df_cables=pd.DataFrame(),
        df_angles=pd.DataFrame(),
        m_per_unit=1.0,
        metadata={},
    )
    graph = build_igraph_graph(graph_input)
    assert graph.ecount() == 2
    assert graph.es["length_m"] == [7.0, 4.0]
    assert graph.es["length_m"][1] == 4.0




