# -*- coding: utf-8 -*-
"""Data Type

  File: 
    /Lancet/core/ops/dtype

  Description: 
    Data Type Operations
"""


import numpy as np


def convert_to_nparray(value, dtype=None):
  """Convert to np.array

    Description:
      FIXME
  """
  if dtype is None:
    return np.array(value)
  return np.array(value, dtype=np.dtype(dtype))

