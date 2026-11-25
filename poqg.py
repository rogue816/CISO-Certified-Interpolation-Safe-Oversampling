"""Deprecated alias for :mod:`ciso`.

The method was named PO-qG while the experiments were run, and the frozen
CSVs under ``results/`` still carry that label. The implementation now lives
in ``ciso.py`` as the class :class:`CISO`. This module exists so that
archived scripts written against ``from poqg import POQG`` keep working.

New code should use::

    from ciso import CISO
"""
import warnings

from ciso import CISO, POQG, g_q, _minmax, _stable_knn_indices, _knn_query  # noqa: F401

warnings.warn(
    "poqg is deprecated; import CISO from the ciso module instead.",
    DeprecationWarning, stacklevel=2,
)

__all__ = ["CISO", "POQG", "g_q"]
