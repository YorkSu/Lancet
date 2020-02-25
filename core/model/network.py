# -*- coding: utf-8 -*-
"""Model

  File: 
    /Lancet/core/model/network

  Description: 
    网络基类
"""


import tensorflow as tf

from Lancet.core.abc import Keras_Network
from Lancet.core.model import layer
from Lancet.core.model import nn
# from Lancet.train import layer
# from Lancet.train import nn


# class Network(Keras_Network):
#   """Network
  
#     Description: 
#       Network V2 from `https://github.com/YorkSu/hat`(r3.0 - alpha)
#       You need to rewrite `build()` method, define and return 
#       a `keras.Model`(or nn.model). If you need to define 
#       parameters, you can rewrite the `args()` method.
#   """
#   def __init__(
#       self,
#       input_shape,
#       output_shape,
#       **kwargs):
#     self.input_shape = input_shape
#     self.output_shape = output_shape
#     self.__dict__ = {**self.__dict__, **kwargs}
#     self.setup()

#   def setup(self):
#     self.args()
#     load_name = '' # config.get('load_name')
#     if load_name:
#       # log.info(f'Load {load_name}', name=__name__)
#       model = tf.keras.models.load_model(load_name)
#     else:
#       model = self.build()
#       if not isinstance(model, tf.keras.models.Model):
#         return None
#         # log.error(f'build() must return `tf.keras.models.Model`, '\
#         #     f'but got {type(model)}', name=__name__, exit=True)
#     self.model = model

#   # method for rewrite

#   def args(self):
#     pass

#   def build(self):
#     # raise NotImplementedError
#     return


class Audio(Keras_Network):
  """Audio
  
    Description: 
      Based on Network
      Network V2 from `https://github.com/YorkSu/hat`(r3.0 - alpha)
      You need to rewrite `build()` method, define and return 
      a `keras.Model`(or nn.model). If you need to define 
      parameters, you can rewrite the `args()` method.
  """
  def __init__(
      self,
      instruments: list,
      input_shape,
      activation='sigmoid',
      output_shape=None,
      **kwargs):
    assert isinstance(instruments, list), f"[Invalid] Audio." \
        f"instruments expect `list`, got {type(instruments)}"
    assert len(instruments) > 0, f"[Invalid] Audio.instruments." \
        f"len must more than `0`, got {len(instruments)}"
    assert isinstance(instruments[0], str), f"[Invalid] Audio."\
        f"instruments.value expect `str`, got {type(instruments[0])}"
    self.instruments = instruments
    self.input_shape = input_shape
    if output_shape is None:
      self.output_shape = input_shape
    else:
      self.output_shape = output_shape
    self.activation = activation
    self.__dict__ = {**self.__dict__, **kwargs}
    self.setup()

  def setup(self):
    self.args()
    load_name = '' # config.get('load_name')
    if load_name:
      # log.info(f'Load {load_name}', name=__name__)
      model = tf.keras.models.load_model(load_name)
    else:
      model = self.multi_inst()
      if not isinstance(model, tf.keras.models.Model):
        return None
        # log.error(f'build() must return `tf.keras.models.Model`, '\
        #     f'but got {type(model)}', name=__name__, exit=True)
    self.model = model

  def multi_inst(self):
    inputs = nn.input(self.input_shape)
    output_channel = []
    for inst in self.instruments:
      output_channel.append(self.build(f"{inst}_Channel")(inputs))
    all_channel = layer.Stack()(output_channel)
    acted_channel = nn.activation(self.activation, name='Activation')(all_channel)
    split_channel = layer.Split()(acted_channel)
    outputs = []
    for inst, c in zip(self.instruments, split_channel):
      outputs.append(nn.multiply(name=f"{inst}_Spectrum")([c, inputs]))
    return nn.model(inputs=inputs, outputs=outputs)

  # method for rewrite

  def args(self):
    pass

  def build(self, name=None):
    # raise NotImplementedError
    def linear(inputs):
      return inputs
    return linear

