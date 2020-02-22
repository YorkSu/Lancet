# -*- coding: utf-8 -*-
"""Array

  File: 
    /Lancet/core/ops/array

  Description: 
    np.Array Operations
"""


import numpy as np


def asarray(value, dtype=None, order=None):
  """Convert to np.array

    Description:
      FIXME
  """
  if dtype is None:
    return np.array(value)
  return np.array(
      value,
      dtype=np.dtype(dtype), 
      copy=False,
      order=order)

