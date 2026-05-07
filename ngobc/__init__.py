"""
NGO-BC: Neural-Guided Online Budget Control for Multi-Agent LLM Systems.

A plug-in budget control layer that sets max_tokens for each agent
before LLM invocation, without modifying agent internals or interaction
topology.

Two-layer architecture:
  - Within-Round (Micro): MPC + Product KKT + Structural Pooling
  - Across-Round (Macro): MWU scheduling with terminal projection
"""

from .kkt import NodeParams, AllocationResult
from .product_kkt import product_kkt_solve
from .parallel_product_kkt import parallel_product_kkt_solve
from .optimizer import PSBCOptimizer
from .psbc_m import PSBCMacroScheduler, compute_node_bids
from .baselines import FixedAllocator, GreedyAllocator
