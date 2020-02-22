# -*- coding: utf-8 -*-
"""Importer

  File: 
    /Lancet/core/lib/importer

  Description: 
    加载器，可动态加载model
"""


import importlib

from Lancet.__config__ import project
from Lancet.core.lib import log


def load(name='', lib=''):
  """Import target class

    Description: 
      FIXME

    Args:
      name: Str. 需要加载的model的名字

    Return:
      hat.Network

    Raises:
      ImportError
  """
  name = str(name)
  lib = lib or name
  target = None

  try:
    module_name = '.'.join([project, lib])  # FIXME
    module = importlib.import_module(module_name)
    target = getattr(module, name)
  except Exception:
    log.error(f"[ImportError] '{name}' not in model.{lib}",
          exit=True, name=__name__)

  if target is None:
    log.error(f"[ImportError] '{name}' not in model.{lib}",
          exit=True, name=__name__)

  return target

