"""Predict."""

from __future__ import annotations
from argparse import Namespace
from logging import (
    basicConfig,
    getLogger,
    INFO,
)
from sys import stdout

from ..micro import NomadScheduled as BaseMicro
from .inputs import Inputs
from .output import Output
from .model import Model


basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=stdout)
logger = getLogger(__name__)  # pylint: disable=invalid-name



class Micro(BaseMicro):
    """Micro."""

    ARGS = {
        **BaseMicro.ARGS,
        **Inputs.ARGS,
        **Output.ARGS,
    }

    DESCRIPTION = 'RIDS Prediction'

    @classmethod
    def from_cfg(cls, cfg: dict) -> Micro:
        """Return micro from cfg."""
        kwargs = {
            key: from_cfg(cfg[key])
            for key, from_cfg in (
                ('input', Inputs.from_cfg),
                ('output', Output.from_cfg),
                ('model', Model.from_cfg),
            )
        }
        return cls(**kwargs)

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> None:
        """Patch cfg from args."""
        for key, patch_args in (
                ('input', Inputs.patch_args),
                ('output', Output.patch_args),
                ('model', Model.patch_args),
        ):
            cfg[key] = patch_args(args, cfg.get(key))
        return cfg

    # @classmethod
    # def replay(cls):
    #     """Main."""
    #     i = cls.from_argv(sys_argv[1:])
    #     print('>>>>i<<<')
    #     print(i)
    #     dt_now = i.inputs.dt_now
    #     for ...
    #         i.inputs.dt_now = modified_now