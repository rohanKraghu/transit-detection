"""Data sources.  Everything here implements :class:`LightCurveSource`.

``loader`` is deliberately **not** re-exported: it depends on
``transitml.features``, which depends on ``transitml.preprocess``, which
depends on this package.  Re-exporting it here would close that cycle and make
``import transitml.preprocess`` fail depending on import order.  Import it
directly instead::

    from transitml.data.loader import build_default_dataset
"""

from .base import LightCurve, LightCurveSource
from .synthetic import SyntheticTESSSource

__all__ = ["LightCurve", "LightCurveSource", "SyntheticTESSSource"]
