# -*- coding: utf-8 -*-
"""Layer

  File: 
    /Lancet/core/model/layer

  Description: 
    自定义网络层
"""


import tensorflow as tf


class Split(tf.keras.layers.Layer):
  """Split
  
    Description:
      FIXME
  """
  def __init__(self, axis=-1, **kwargs):
    super().__init__(**kwargs)
    self.axis = axis

  def build(self, input_shape):
    self.built = True

  def call(self, inputs):
    return 

  def get_config(self):
    config = {
        'axis': self.axis,}
    return dict(list(super().get_config().items()) + list(config.items()))

