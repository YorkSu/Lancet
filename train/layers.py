# -*- coding: utf-8 -*-
"""Layers

  File: 
    /Lancet/train/layers

  Description: 
    自定义网络层
"""


import tensorflow as tf
import librosa as lrs


# ========================
# Librosa transform
# ========================


class ToRMS(tf.keras.layers.Layer):
  """ToRMS
  
    Description:
      Based on librosa.feature.rms

    Args:
      None

    Input:
      tf.Tensor, shape=(None, 1D)

    Return:
      tf.Tensor, shape=(None, 1D)
  """
  def __init__(self, **kwargs):
    super().__init__(trainable=False, **kwargs)
  def np_rms(self, y):
    return lrs.feature.rms(y=y)
  @tf.function
  def tf_rms(self, y):
    return tf.numpy_function(self.np_rms, [y], tf.float32)
  def call(self, inputs):
    outputs = self.tf_rms(inputs)
    return tf.keras.layers.Flatten()(outputs)



