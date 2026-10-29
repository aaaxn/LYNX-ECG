"""Hierarchical GNN model for beat-graph ECG classification.

Two-level architecture:

- ``SubgraphEncoder`` (GINE): operates on per-beat visibility-graph subgraphs.
  Each subgraph is one beat with K nodes (default 30) and 2 node features
  (``amplitude``, ``t_norm_beat``) and 2 edge attributes (``delta_t``,
  ``delta_y``). Produces one embedding per subgraph.
- ``SupergraphEncoder`` (GATv2): operates on the per-(exam, lead) chain of
  subgraph embeddings. Supergraph edges are sequential (and optionally skip)
  edges with a signed integer ``edge_step`` that becomes a learned bias via
  bucket embedding. Each lead's chain is independent, with no cross-lead edges.

Aggregation: ``AttentionalAggregation`` pools subgraphs into per-lead vectors,
then a second attention pools leads into a single per-exam vector. The
``ClassificationHead`` projects to ``num_classes`` logits for BCEWithLogits
multi-label training.
"""

from models.heads import ClassificationHead
from models.hier_model import HierGNNModel
from models.subgraph_encoder import SubgraphEncoder
from models.supergraph_encoder import SupergraphEncoder

__all__ = [
    "ClassificationHead",
    "HierGNNModel",
    "SubgraphEncoder",
    "SupergraphEncoder",
]
