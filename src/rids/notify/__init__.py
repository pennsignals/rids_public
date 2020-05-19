"""Notify."""

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
        **Model.ARGS,
        **Output.ARGS,
    }

    DESCRIPTION = 'RIDS Notification'

    @classmethod
    def from_cfg(cls, cfg: dict) -> Micro:
        """Return micro from cfg."""
        print("**in __init__py Micro from_cfg")
        # E.g. key:value :: 'input':Inputs.from_cfg(cfg['input'])

        Model.from_cfg(cfg['model'])
        kwargs = {key: from_cfg_function(cfg[key]) for key, from_cfg_function in (('input', Inputs.from_cfg),
                                                                                  ('output', Output.from_cfg),
                                                                                  ('model', Model.from_cfg),)
                  }
        return cls(**kwargs)

    @classmethod
    def patch_args(cls, args: Namespace, cfg: dict) -> dict:
        """Patch cfg from args."""
        print("**in __init__py Micro patch_args")
        # print(cfg['model'])
        # E.g. cfg[key] = patch_args_function(args, cfg.get(key))
        # E.g. cfg['model'] = Model.patch_args(args, cfg['model'])
        #
        # print('>>>WTF')
        # cfg['input'] = Inputs.patch_args(args, cfg=cfg['input'])
        # cfg['model'] = Model.patch_args(args, cfg=cfg['model'])
        # print('<<<WTF')
        # # Passing argparse.Namespace instead of model.Model. Is this intentional?
        # # Parameter args unfilled
        #
        for key, patch_args_function in (('input', Inputs.patch_args),
                                         ('output', Output.patch_args),
                                         ('model', Model.patch_args),
                                         ):
            print('...'+key)
            print('...args<<<')
            print([(key2, len(str(value2))) for key2, value2 in vars(args).items()])
            print('...cfg<<<')
            print(cfg.get(key))
            cfg[key] = patch_args_function(args, cfg.get(key))
        return cfg
