"""HQAP: nachvollziehbare Bausteine für Kapitel 3 von arXiv:2510.25838v1.

Zeitliche Reihenfolge ist IMMER links nach rechts: R.then(P) bedeutet P @ R.
Die öffentlich nutzbaren Zwischenschritte liegen in protocol.build().
"""
from .circuit import Circuit, Gate
from .topology import heavy_hex
from .protocol import Config, build

__all__ = ["Circuit", "Gate", "Config", "build", "heavy_hex"]
