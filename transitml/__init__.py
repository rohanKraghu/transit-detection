"""Transit detection on astronomical light curves.

A small, end-to-end machine-learning pipeline: generate TESS-like photometry,
detrend it, run a Box Least Squares search, turn the search output into vetting
features, and classify planets against eclipsing binaries and noise under
severe class imbalance.
"""

from .config import Config, default_config

__all__ = ["Config", "default_config"]
__version__ = "0.1.0"
