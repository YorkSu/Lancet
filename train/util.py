# -*- coding: utf-8 -*-
"""Util

  File: 
    /Lancet/train/util

  Description: 
    utils
"""


class Counter(object):
  """Counter 
    
    Description: 
      全局计数器，根据名字自动计数

    Args:
      name: Str. 计数器名字

    Return:
      Str number.

    Usage:
      The `Counter` can be used as a `str` object.
      Or like a function. 
      This two method use the `default initial value 1`.
      And each visit will automatically increase by 1.
  """
  count = {}

  def __init__(self, name=None):
    self.name = name

  def __str__(self):
    if self.name:
      if self.name not in Counter.count:
        Counter.count[self.name] = 1
      else:
        Counter.count[self.name] += 1
      return str(Counter.count[self.name])
    return None

