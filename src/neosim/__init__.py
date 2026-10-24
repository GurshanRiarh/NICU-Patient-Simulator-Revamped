"""NeoSim 2.0: a generative, event-aware NICU patient simulator."""

__version__ = "2.0.0"

from neosim.config import CONDITIONS, EVENT_TYPES, SIGNALS, SimulationConfig

__all__ = ["CONDITIONS", "EVENT_TYPES", "SIGNALS", "SimulationConfig", "__version__"]
