# -*- coding: utf-8 -*-
"""Lancet Config

  File:
    /Lancet/__config__

  Description:
    Lancet config
"""


# ================
# Import
# ================


from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function
import os as _os
import sys as _sys


Lancet_path = _os.path.dirname(_os.path.abspath(__file__))
if Lancet_path not in _sys.path:
  _sys.path.append(Lancet_path)


del _os, _sys, _absolute_import, _division, _print_function


# ================
# Config
# ================


project = 'Lancet'
author = ["York Su", "Vickko"]
github = "https://github.com/YorkSu/Lancet"
release = False
short_version = '1.0'
full_version = '1.0 - alpha'
version = short_version if release else full_version
full_name = f'{project} v{version}'
root = Lancet_path


__version__ = version
__root__ = Lancet_path




