# -*- coding: utf-8 -*-
"""Layers

  File: 
    /Lancet/train/layers

  Description: 
    自定义网络层
"""


import numpy as np
import tensorflow as tf
import librosa as lrs


# ========================
# Librosa transform
# ========================


class ToStft(tf.keras.layers.Layer):
  """ToStft
  
    Description:
      Based on librosa.stft

    Args:
      n_fft: Int, default 2048.
      hop_length: Int, default None. None -> win_length/4
      win_length: Int, default None. None -> n_fft
      window: Str, default 'hann'.
      center: Bool, default True.
      dtype: Str, default None. None -> 'float32'
      pad_mode: Str, default 'reflect'.

    Input:
      tf.Tensor, shape=(None, Sample, Channel)

    Return:
      tf.Tensor, shape=(None, floor(1+n_fft/2), ceil(Sample/hop_length), Channel)
  """
  def __init__(
      self,
      n_fft=2048,
      hop_length=None,
      win_length=None,
      window='hann',
      center=True,
      dtype=None,
      pad_mode='reflect',
      **kwargs):
    super().__init__(trainable=False, **kwargs)
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    self.window = window
    self.center = center
    if dtype is None:
      self.np_dtype = np.float32
      self.tf_dtype = tf.float32
      # FIXME: 换成识别全局dtype字符串然后自动选择
    else:
      self.np_dtype = dtype
      self.tf_dtype = dtype
    self.pad_mode = pad_mode
  def np_stft(self, y):
    outputs = []
    for i in range(y.shape[1]):
      outputs.append(lrs.stft(
        y=np.asfortranarray(y[:, i]),
        n_fft=self.n_fft,
        hop_length=self.hop_length,
        win_length=self.win_length,
        window=self.window,
        center=self.center,
        dtype=self.np_dtype,
        pad_mode=self.pad_mode)[:, :, np.newaxis])
    return np.concatenate(outputs, axis=2)
  @tf.function
  def tf_stft(self, y):
    return tf.numpy_function(self.np_stft, [y], self.tf_dtype)
  def call(self, inputs):
    outputs = inputs
    input_shape = tf.keras.backend.int_shape(inputs)
    assert len(input_shape) in [2, 3], f"The legal nD of input " \
        f"shape of {ToStft} must be 2 or 3, but got {len(input_shape)}"
    if len(input_shape) == 2:
      outputs = self.tf_stft(outputs)
    if len(input_shape) == 3:
      temp = []
      for i in range(input_shape[0]):
        xi = tf.keras.layers.Lambda(
            lambda x, t: x[t, :, :],
            output_shape=input_shape[1:],
            arguments={'t': i})(outputs)
        yi = self.tf_stft(xi)
        temp.append(tf.expand_dims(yi, 0))
      outputs = tf.keras.layers.Concatenate(axis=0)(temp)
    return outputs

if __name__ == "__main__":
  filepath = './dataset/york/1_Vox.wav'
  y, sr = lrs.load(filepath, sr=None, mono=False)
  y = y.transpose(1, 0)
  y = tf.Variable([y, y])
  print(y.shape)
  y = ToStft()(y)
  print(y.shape)


