# -*- coding: utf-8 -*-
"""Lancet
  =====

  Version:
    1.0 - alpha

  Description:
    # FIXME
    None

  Requirements:
    # FIXME
    App part:
      None
    Train part:
      Tensorflow 2.0 (or 1.14+)(gpu recommended)
      CUDA 10.0.130 (if gpu)
      cuDNN 7.6.5 (if gpu)

  License:
    Copyright 2020 The HAT Authors. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
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
if __root__ not in _sys.path:
  _sys.path.append(Lancet_path)


# ================
# Config
# ================


project = 'Lancet'
author = ["York Su", "Vickko"]
github = "https://github.com/YorkSu/Lancet"
release = False
short_version = '1.0'
full_version = '1.0 - alpha'
version = release and short_version or full_version
full_name = f'{project} v{version}'
root = path
__version__ = version
__root__ = path


# del _os, _sys, _absolute_import, _division, _print_function

