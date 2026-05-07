"""
psbc/dag.py – DAG utilities for PSBC single-round static topology.

Provides:
  - topological_sort(adj)          → list[str] in topo order
  - get_descendants(adj, node)     → set[str]
  - get_ancestors(adj, node)       → set[str]
  - critical_path_nodes(adj)       → set[str]  (nodes on any longest path)
  - topo_weights(adj)              → dict[str, float]  (w_j, sums to 1)

adj: Dict[str, List[str]]  adjacency list, adj[u] = [v, ...] means u→v
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_in_degree(adj: Dict[str, List[str]]) -> Dict[str, int]:
    in_deg: Dict[str, int] = {n: 0 for n in adj}
    for u, children in adj.items():
        for v in children:
            in_deg.setdefault(v, 0)
            in_deg[v] += 1
    return in_deg


def _all_nodes(adj: Dict[str, List[str]]) -> Set[str]:
    nodes: Set[str] = set(adj.keys())
    for children in adj.values():
        nodes.update(children)
    return nodes


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def topological_sort(adj: Dict[str, List[str]]) -> List[str]:
    """
    Return a list of all nodes in a valid topological order (Kahn's algorithm).
    Raises ValueError if the graph contains a cycle.
    """
    nodes = _all_nodes(adj)
    in_deg: Dict[str, int] = {n: 0 for n in nodes}
    for u in adj:
        for v in adj[u]:
            in_deg[v] += 1

    queue = deque(n for n in nodes if in_deg[n] == 0)
    order: List[str] = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj.get(u, []):
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    if len(order) != len(nodes):
        raise ValueError("Graph contains a cycle; topological sort is undefined.")
    return order


def get_descendants(adj: Dict[str, List[str]], node: str) -> Set[str]:
    """Return all nodes reachable from `node` (excluding `node` itself)."""
    visited: Set[str] = set()
    stack = list(adj.get(node, []))
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            stack.extend(adj.get(v, []))
    return visited


def get_ancestors(adj: Dict[str, List[str]], node: str) -> Set[str]:
    """Return all nodes that can reach `node` (excluding `node` itself)."""
    # Build reverse adjacency
    rev: Dict[str, List[str]] = defaultdict(list)
    for u, children in adj.items():
        for v in children:
            rev[v].append(u)
    return get_descendants(rev, node)


def _sinks(adj: Dict[str, List[str]]) -> Set[str]:
    """Return nodes with no outgoing edges (leaf / sink nodes)."""
    nodes = _all_nodes(adj)
    has_children = {u for u, ch in adj.items() if ch}
    return nodes - has_children


def critical_path_nodes(adj: Dict[str, List[str]]) -> Set[str]:
    """
    Return the set of nodes that lie on at least one longest path
    (measured in number of nodes, i.e. hop count).

    These are the nodes PSBC treats as critical: if they cannot be
    executed, the whole round is terminated rather than skipped.
    """
    order = topological_sort(adj)
    nodes = _all_nodes(adj)

    # Forward pass: longest path length ending at each node (inclusive)
    dist: Dict[str, int] = {n: 1 for n in nodes}
    for u in order:
        for v in adj.get(u, []):
            if dist[u] + 1 > dist[v]:
                dist[v] = dist[u] + 1

    max_len = max(dist.values(), default=0)

    # Backward pass: longest path length starting at each node (inclusive)
    rev: Dict[str, List[str]] = defaultdict(list)
    for u, children in adj.items():
        for v in children:
            rev[v].append(u)

    rev_dist: Dict[str, int] = {n: 1 for n in nodes}
    for u in reversed(order):
        for v in rev.get(u, []):
            if rev_dist[u] + 1 > rev_dist[v]:
                rev_dist[v] = rev_dist[u] + 1

    # A node is on a longest path iff dist[v] + rev_dist[v] - 1 == max_len
    return {v for v in nodes if dist[v] + rev_dist[v] - 1 == max_len}


def topo_weights(adj: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute normalised topological importance weights w_j for every node.

    w_j^raw = number of sink nodes reachable from j  (sinks count themselves)
    w_j     = w_j^raw / sum_k(w_k^raw)

    Returns a dict {node_id: weight} that sums to 1.0.
    If the graph has only one node, that node gets weight 1.0.
    """
    nodes = _all_nodes(adj)
    sink_set = _sinks(adj)

    raw: Dict[str, int] = {}
    for node in nodes:
        reachable = get_descendants(adj, node) | {node}
        raw[node] = len(reachable & sink_set)
        # Ensure at least 1 so no node gets weight 0
        if raw[node] == 0:
            raw[node] = 1

    total = sum(raw.values())
    return {n: raw[n] / total for n in nodes}
