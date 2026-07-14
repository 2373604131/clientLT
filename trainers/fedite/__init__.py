"""FedITE: Federated Internal Tail Evidence Adaptation.

FedITE is intentionally implemented as an independent method path. It does
not modify CAPT, FedTEF, or their trainers. In FedITE, "topology" means
class-client evidence topology, not physical communication topology.
"""

from .aggregation import aggregate_fedite
from .model import FedITEModel
from .observer import EvidenceTopologyObserver
from .trainer import FedITEClientTrainer

__all__ = [
    "FedITEModel",
    "FedITEClientTrainer",
    "EvidenceTopologyObserver",
    "aggregate_fedite",
]
