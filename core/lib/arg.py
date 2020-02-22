# -*- coding: utf-8 -*-
"""Arg

  File: 
    /Lancet/core/lib/arg

  Description: 
    Global Arguments Manager
"""


import argparse

from Lancet.core.abc import ClassA


# ========================
# Config
# ========================


_ARGS_MAP = {
    
    }


# ========================
# Functions
# ========================


# ========================
# Arguments Manager
# ========================


class _Argument(ClassA):
  """_Argument

    Description:
      FIXME
  """
  def __init__(self, *args, **kwargs):
    # ========================
    # Default Parameters
    # ========================
    self.model_name = ''
    self.batch_size = 0
    self.epochs = 0
    self.step = 0
    self.step_per_log = 0
    self.step_per_val = 0
    self.opt = ''
    self.loss = ''
    self.metrics = []
    self.dtype = ''
    self.run_mode = ''
    self.addition = ''
    self.aug = None
    # ========================
    # Empty Parameters
    # ========================
    self._argv = ''
    del args, kwargs

  def init(self, argv: str):
    if argv:
      self._argv = argv
    else:
      self._argv = input("=>")
    self._parse()

  def _parse(self):
    parser = argparse.ArgumentParser()

    # set
    parser.add_argument('-m', '-model', dest='model_name', type=str, action="store_true")
    parser.add_argument('-b', '-batch_size', dest='batch_size', type=int, action="store_true")
    parser.add_argument('-e', '-epoch', dest='epoch', type=int, action="store_true")
    parser.add_argument('-s', '-step', dest='step', type=int, action="store_true")
    parser.add_argument('-a', '-addition', dest='addition', type=str, action="store_true")
    parser.add_argument('-t', '-type', dest='dtype', type=str, action="store_true")
    parser.add_argument('-sl', '-step_per_log', dest='step_per_log', type=int, action="store_true")
    parser.add_argument('-sv', '-step_per_val', dest='step_per_val', type=int, action="store_true")

    # tag
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--g', '--get_image', dest='run_mode', const='get_image', default='', action='store_const')
    group1.add_argument('--t', '--train_only', dest='run_mode', const='train_only', default='', action='store_const')
    group1.add_argument('--v', '--val_only', dest='run_mode', const='val_only', default='', action='store_const')
    
    args, unparsed = parser.parse_known_args(self._argv.split())
    self.__dict__ = {**self.__dict__, **args.__dict__}
    # log unparsed, unparsed is list of strings
    # 参见hat.config
    del unparsed
      

_ARG = _Argument()


def init(argv: str):
  _ARG.init(argv)

def get(name, default=None):
  return _ARG.get(name, default)

def set(name, value):  # pylint:disable=redefined-builtin
  _ARG.set(name, value)


