"""Configurable."""

from __future__ import annotations
from argparse import Namespace


class Configurable:
    """Configurable."""

    @classmethod
    def from_cfg(  # pylint: disable=unused-argument
            cls,
            cfg: dict) -> Configurable:
        """Return model from cfg."""
        return cls()

    @classmethod
    def patch_args(  # pylint: disable=unused-argument
            cls, args: Namespace,
            cfg: dict) -> dict:
        """Patch args into cfg."""
        return cfg