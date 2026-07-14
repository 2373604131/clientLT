"""Standalone TCRM-Core components.

TCRM deliberately does not register a Dassl trainer and does not depend on
federated_main.py. The public entrypoint is tcrm_main.py.
"""

from .state import TCRMCoreState

__all__ = ["TCRMCoreState"]
