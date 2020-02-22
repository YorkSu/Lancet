# -*- coding: utf-8 -*-
"""Model

  File: 
    /Lancet/core/model/network

  Description: 
    网络基类
"""


import tensorflow as tf

from Lancet.core.abc import Keras_Network
# from Lancet.train import layer
# from Lancet.train import nn


class Network(Keras_Network):
  """Network
  
    Description: 
      Network V2 from `https://github.com/YorkSu/hat`(r3.0 - alpha)
      You need to rewrite `build()` method, define and return 
      a `keras.Model`(or nn.model). If you need to define 
      parameters, you can rewrite the `args()` method.
  """
  def __init__(
      self,
      input_shape,
      output_shape,
      **kwargs):
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.setup()
    del kwargs

  def setup(self):
    self.args()
    load_name = '' # config.get('load_name')
    if load_name:
      # log.info(f'Load {load_name}', name=__name__)
      model = tf.keras.models.load_model(load_name)
    else:
      model = self.build()
      if not isinstance(model, tf.keras.models.Model):
        return None
        # log.error(f'build() must return `tf.keras.models.Model`, '\
        #     f'but got {type(model)}', name=__name__, exit=True)
    self.model = model

  # method for rewrite

  def args(self):
    pass

  def build(self):
    # raise NotImplementedError
    return

