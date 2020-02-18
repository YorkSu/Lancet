# -*- coding: utf-8 -*-
"""Model

  File: 
    /Lancet/train/model

  Description: 
    模型库
"""


import tensorflow as tf

from Lancet.train import abc
from Lancet.train import layer
from Lancet.train import nn


class Network(abc.Keras_Network):
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


class UNet(Network):
  """UNet

    Description:
      None
  """
  def args(self):
    pass

  def build(self):
    inputs = nn.input(self.input_shape)
    x = inputs
    x = layer.Stft()(x)
    # x = self.nn.dense(self.output_shape, activation='softmax')(x)
    x = layer.Istft(length=self.input_shape[0])(x)
    return nn.model(inputs, x)


class Blade10(Network):
  """Bladel10  
  
    Description:
      None
  """
  def args(self):
    pass

  def build(self):
    inputs = nn.input(self.input_shape)
    x = inputs
    # x = self.nn.dense(self.output_shape, activation='softmax')(x)
    return nn.model(inputs, x)


if __name__ == "__main__":
  model = UNet((512 * 512, 2), (512 * 512, 2))
  model.summary()



