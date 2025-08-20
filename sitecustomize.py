"""Runtime compatibility shims.

This module is auto-imported by Python (if present) after the standard site
initialization. We use it to provide backward-compatible NumPy aliases that
some third-party libraries (that haven't yet been updated for NumPy 2.x) may
still reference (e.g. np.float_, np.int_, etc.). If the deployment environment
installs NumPy 2 despite our <2.0 pin (e.g. due to a cached layer or preinstalled
package), these aliases prevent AttributeError crashes.
"""
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

try:
    np = importlib.import_module("numpy")  # type: ignore
except Exception as e:  # pragma: no cover
    logger.warning("sitecustomize: NumPy not available: %s", e)
else:  # Only run shims if NumPy imported
    # Map of deprecated alias names to the modern dtype objects.
    alias_map = {
        "float_": getattr(np, "float64", None),
        "float": getattr(np, "float64", None),
        "int_": getattr(np, "int64", None),
        "int": getattr(np, "int64", None),
        "bool_": getattr(np, "bool_", None),
        "object_": getattr(np, "object_", None),
    }
    created = []
    for name, target in alias_map.items():
        if target is None:
            continue
        if not hasattr(np, name):
            try:
                setattr(np, name, target)
                created.append(name)
            except Exception as e:  # pragma: no cover
                logger.debug("sitecustomize: failed to set numpy.%s: %s", name, e)
    if created:
        logger.info("sitecustomize: added NumPy aliases: %s", ", ".join(created))
