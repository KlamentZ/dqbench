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


def build_qaoa_circuit(
    graph: list[tuple[int, int]],
    n_qubits: int,
    p_layers: int,
) -> dq.QubitCircuit:
    """Build a parameterized QAOA Max-Cut ansatz circuit.

    Constructs the standard QAOA ansatz:
    ``|ψ(γ,β)⟩ = U_B(β_p) U_C(γ_p) ... U_B(β_1) U_C(γ_1) |+⟩^n``

    Each gate is independently parameterized via ``encode=True``, giving
    ``p_layers * (len(graph) + n_qubits)`` total parameters. The caller
    passes a flat tensor to ``circuit(params)`` at execution time.

    The cost unitary U_C uses the CNOT–RZ–CNOT decomposition for each edge,
    implementing ``exp(-iθ Z_u Z_v / 2)``. To recover the standard QAOA
    phase ``exp(-iγ Z_u Z_v)``, pass ``2γ`` as the rotation angle.

    ZZ observables are added for every edge so that ``circuit.expectation()``
    returns ``⟨Z_u Z_v⟩`` per edge after a forward pass.

    Args:
        graph: Edge list as sorted (u, v) tuples with u < v.
        n_qubits: Number of qubits (= number of graph vertices).
        p_layers: Number of QAOA layers (depth of the ansatz).

    Returns:
        A DeepQuantum QubitCircuit with encode=True parameters and
        ZZ observables registered on every edge.
    """
    cir = dq.QubitCircuit(n_qubits)

    cir.hlayer()

    for _ in range(p_layers):
        for u, v in graph:
            cir.cnot(u, v)
            cir.rz(v, encode=True)
            cir.cnot(u, v)
        cir.barrier()
        for q in range(n_qubits):
            cir.rx(q, encode=True)
        cir.barrier()

    for u, v in graph:
        cir.observable(wires=[u, v], basis='z')

    return cir


def generate(
    n_qubits: list[int],
    p_layers: list[int],
    graph_type: str,
    graph_params: dict[str, Any],
    n_seeds: int,
    seed: int,
) -> list[QAOAInstance]:
    """Generate a sweep of QAOA Max-Cut benchmark instances.

    Iterates the cross-product ``n_qubits × p_layers × range(n_seeds)`` and
    builds one ``QAOAInstance`` per combination. Iteration order is
    ``n_qubits`` (outer) → ``p_layers`` → seed index (inner), so the returned
    list groups first by problem size, then by ansatz depth, then by seed.

    Per-instance seeds are derived as ``seed + s_idx`` for ``s_idx`` in
    ``range(n_seeds)``. This linear scheme keeps reproduction trivial: the
    same ``(seed, n_seeds)`` always yields the same sequence of per-instance
    seeds, and inspecting ``instance.seed`` directly reveals which run it is.

    Note: the same ``(n, instance_seed)`` graph is regenerated for every
    ``p`` in ``p_layers``, and its brute-force max-cut is recomputed each
    time. This is wasteful but acceptable for Phase 1 (n_qubits ≤ 20).
    ``graph_params`` is shared by reference across all returned instances;
    callers must not mutate it.

    Args:
        n_qubits: Problem sizes to sweep, e.g. ``[6, 8, 10]``.
        p_layers: QAOA depths to sweep, e.g. ``[1, 2, 3]``.
        graph_type: Graph family passed through to ``_generate_graph``.
        graph_params: Type-specific graph parameters, e.g.
            ``{'p_edge': 0.5}`` for ``'erdos_renyi'``.
        n_seeds: Number of seeds per ``(n, p)`` combination. Must be >= 1.
        seed: Base seed; per-instance seeds are ``seed, seed+1, ...``.

    Returns:
        A list of ``QAOAInstance`` of length
        ``len(n_qubits) * len(p_layers) * n_seeds``.

    Raises:
        ValueError: If ``n_seeds < 1``, or propagated from ``_generate_graph``
            when ``graph_type`` / ``graph_params`` are invalid.

    Example:
        >>> instances = generate(
        ...     n_qubits=[6],
        ...     p_layers=[1, 2],
        ...     graph_type='erdos_renyi',
        ...     graph_params={'p_edge': 0.5},
        ...     n_seeds=3,
        ...     seed=42,
        ... )
        >>> len(instances)
        6
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")

    instances: list[QAOAInstance] = []
    for n in n_qubits:
        for p in p_layers:
            for s_idx in range(n_seeds):
                instance_seed = seed + s_idx
                graph = _generate_graph(graph_type, graph_params, n, instance_seed)
                circuit = build_qaoa_circuit(graph, n, p)
                classical_max_cut = _compute_max_cut_brute_force(graph, n)
                instances.append(QAOAInstance(
                    circuit=circuit,
                    graph=graph,
                    n_qubits=n,
                    p_layers=p,
                    graph_type=graph_type,
                    graph_params=graph_params,
                    seed=instance_seed,
                    classical_max_cut=classical_max_cut,
                ))
    return instances
