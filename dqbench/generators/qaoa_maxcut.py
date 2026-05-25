"""QAOA Max-Cut circuit generator.

This module produces parameterized QAOA Max-Cut benchmark instances. Each
instance bundles a DeepQuantum circuit, the underlying graph, and metadata
needed for reproducibility and evaluation.
"""

from dataclasses import dataclass
from typing import Any

import deepquantum as dq
import networkx as nx


@dataclass
class QAOAInstance:
    """A single QAOA Max-Cut benchmark instance.

    Bundles a parameterized DeepQuantum circuit with the underlying graph
    and metadata needed to reproduce, identify, and evaluate this instance.

    Attributes:
        circuit: A DeepQuantum QubitCircuit with encode=True parameters.
        graph: List of (vertex_i, vertex_j) tuples representing graph edges.
        n_qubits: Number of qubits (= number of vertices in the graph).
        p_layers: Number of QAOA layers.
        graph_type: Family the graph was sampled from, e.g. 'erdos_renyi'.
        graph_params: Parameters used to sample this graph.
        seed: Random seed used for reproducibility.
        classical_max_cut: Exact max-cut via brute force (the optimal reference).
    """

    circuit: dq.QubitCircuit
    graph: list[tuple[int, int]]
    n_qubits: int
    p_layers: int
    graph_type: str
    graph_params: dict[str, Any]
    seed: int
    classical_max_cut: int

    def __repr__(self) -> str:
        return (
            f"QAOAInstance(n_qubits={self.n_qubits}, "
            f"p_layers={self.p_layers}, "
            f"graph_type={self.graph_type!r}, "
            f"n_edges={len(self.graph)}, "
            f"classical_max_cut={self.classical_max_cut}, "
            f"seed={self.seed})"
        )
def _compute_max_cut_brute_force(graph: list[tuple[int, int]], n_qubits: int) -> int:
    """Compute the maximum cut value by brute-force enumeration.

    Iterates over all 2**n_qubits possible vertex partitions, counts cut edges
    for each, and returns the maximum. Tractable for n_qubits <= 20.

    Args:
        graph: Edge list as (vertex_i, vertex_j) tuples.
        n_qubits: Number of vertices in the graph. Passed explicitly because
            graph alone may not reveal isolated vertices.

    Returns:
        The maximum number of cut edges over all 2**n_qubits partitions.
    """
    max_cut = 0
    for assignment in range(2 ** n_qubits):
        cut = sum(
            1 for u, v in graph
            if ((assignment >> u) & 1) != ((assignment >> v) & 1)
        )
        if cut > max_cut:
            max_cut = cut
    return max_cut


def _generate_graph(
    graph_type: str,
    graph_params: dict[str, Any],
    n_qubits: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Generate a graph of the specified family.

    Currently supports only 'erdos_renyi'. The function dispatches on
    graph_type; adding a new graph type means adding a new branch and
    documenting its expected graph_params keys.

    Args:
        graph_type: Currently must be 'erdos_renyi'.
        graph_params: Type-specific parameters.
            For 'erdos_renyi': requires 'p_edge' (float in [0, 1]).
        n_qubits: Number of vertices in the graph.
        seed: Random seed for reproducibility.

    Returns:
        Edge list as sorted tuples (u, v) with u < v, for canonical form.

    Raises:
        ValueError: If graph_type is unsupported or required graph_params keys
            are missing.
    """
    if graph_type == 'erdos_renyi':
        if 'p_edge' not in graph_params:
            raise ValueError(
                "graph_type='erdos_renyi' requires 'p_edge' in graph_params, "
                f"got {graph_params!r}"
            )
        nx_graph = nx.erdos_renyi_graph(
            n=n_qubits,
            p=graph_params['p_edge'],
            seed=seed,
        )
        return [(min(u, v), max(u, v)) for u, v in nx_graph.edges()]

    raise ValueError(
        f"Unsupported graph_type: {graph_type!r}. "
        "Currently supported: 'erdos_renyi'."
    )
