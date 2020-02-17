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


_DTYPE = 'float32'
np_dtype = np.float32
tf_dtype = tf.float32


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
    """Numpy stft"""
    outputs = []
    for i in range(y.shape[1]):
      xi = lrs.stft(
          y=np.asfortranarray(y[:, i]),
          n_fft=self.n_fft,
          hop_length=self.hop_length,
          win_length=self.win_length,
          window=self.window,
          center=self.center,
          pad_mode=self.pad_mode)
      # stft变换得到复数(complex)
      # 通过abs函数求复数与共轭复数乘积的平方根
      yi = np.abs(xi)[:, :, np.newaxis].astype(self.np_dtype)
      outputs.append(yi)
    return np.concatenate(outputs, axis=2)
  @tf.function
  def tf_stft(self, y):
    """TF stft"""
    return tf.numpy_function(self.np_stft, [y], self.tf_dtype)
  def call(self, inputs):
    outputs = inputs
    input_shape = tf.keras.backend.int_shape(inputs)
    assert len(input_shape) in [2, 3], f"The legal nD of input " \
        f"shape of `ToStft` must be 2 or 3, but got {len(input_shape)}"
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


class ToMel(tf.keras.layers.Layer):
  """ToMel
    
    Description:
      Based on librosa.feature.melspectrogram

    Args:
      sr: Int. Sample rate.
      n_mels: Int, default 128.
      n_fft: Int, default 2048.
      hop_length: Int, default None. None -> win_length/4
      win_length: Int, default None. None -> n_fft
      window: Str, default 'hann'.
      center: Bool, default True.
      dtype: Str, default None. None -> 'float32'
      pad_mode: Str, default 'reflect'.
      power: Float, default 2.0.

    Input:
      tf.Tensor, shape=(None, Sample, Channel)

    Return:
      tf.Tensor, shape=(None, n_mels, ceil(Sample/hop_length), Channel)
  """
  def __init__(
      self,
      sr,
      n_mels=128,
      n_fft=2048,
      hop_length=None,
      win_length=None,
      window='hann',
      center=True,
      dtype=None,
      pad_mode='reflect',
      power=2.0,
      **kwargs):
    super().__init__(trainable=False, **kwargs)
    self.sr = sr
    self.n_mels = n_mels
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
    self.power = power
  def np_mel(self, y):
    """Numpy mel"""
    outputs = []
    for i in range(y.shape[1]):
      outputs.append(lrs.feature.melspectrogram(
          y=np.asfortranarray(y[:, i]),
          sr=self.sr,
          S=None,
          n_mels=self.n_mels,
          n_fft=self.n_fft,
          hop_length=self.hop_length,
          win_length=self.win_length,
          window=self.window,
          center=self.center,
          pad_mode=self.pad_mode,
          power=self.power)[:, :, np.newaxis])
    return np.concatenate(outputs, axis=2)
  @tf.function
  def tf_mel(self, y):
    return tf.numpy_function(self.np_mel, [y], self.tf_dtype)
  def call(self, inputs):
    outputs = inputs
    input_shape = tf.keras.backend.int_shape(inputs)
    assert len(input_shape) in [2, 3], f"The legal nD of input " \
        f"shape of `ToMel` must be 2 or 3, but got {len(input_shape)}"
    if len(input_shape) == 2:
      outputs = self.tf_mel(outputs)
    if len(input_shape) == 3:
      temp = []
      for i in range(input_shape[0]):
        xi = tf.keras.layers.Lambda(
            lambda x, t: x[t, :, :],
            output_shape=input_shape[1:],
            arguments={'t': i})(outputs)
        yi = self.tf_mel(xi)
        temp.append(tf.expand_dims(yi, 0))
      outputs = tf.keras.layers.Concatenate(axis=0)(temp)
    return outputs


class ToCqt(tf.keras.layers.Layer):
  """ToCqt
  
    Description:
      Based on librosa.cqt

    Args:
      sr: Int. Sample rate.
      n_mels: Int, default 128.
      n_fft: Int, default 2048.
      hop_length: Int, default None. None -> win_length/4
      win_length: Int, default None. None -> n_fft
      window: Str, default 'hann'.
      center: Bool, default True.
      dtype: Str, default None. None -> 'float32'
      pad_mode: Str, default 'reflect'.
      power: Float, default 2.0.

    Input:
      tf.Tensor, shape=(None, Sample, Channel)

    Return:
      tf.Tensor, shape=(None, n_mels, ceil(Sample/hop_length), Channel)
  """
  def __init__(
      self,
      mode,
      sr,
      n_bins=84,
      hop_length=512,
      fmin=None,
      bins_per_octave=12,
      tuning=0.0,
      filter_scale=1,
      norm=1,
      sparsity=0.01,
      window='hann',
      scale=True,
      pad_mode='reflect',
      dtype=None,
      **kwargs):
    super().__init__(trainable=False, **kwargs)
    self.mode = mode
    self.sr = sr
    self.n_bins = n_bins
    self.hop_length = hop_length
    self.fmin = fmin
    self.bins_per_octave = bins_per_octave
    self.tuning = tuning
    self.filter_scale = filter_scale
    self.norm = norm
    self.sparsity = sparsity
    self.window = window
    self.scale = scale
    self.pad_mode = pad_mode
    if dtype is None:
      self.np_dtype = np.float32
      self.tf_dtype = tf.float32
      # FIXME: 换成识别全局dtype字符串然后自动选择
    else:
      self.np_dtype = dtype
      self.tf_dtype = dtype
  def np_cqt(self, y):
    """Numpy cqt"""
    if self.mode == 'cqt':
      method = lrs.cqt
    elif self.mode in ['hcqt', 'hybrid-cqt']:
      method = lrs.hybrid_cqt
    elif self.mode in ['pcqt', 'pseudo-cqt']:
      method = lrs.pseudo_cqt
    else:
      raise Exception(f"[ModeError] ToCqt.mode illegal, got {self.mode}")
    outputs = []
    for i in range(y.shape[1]):
      xi = method(
          y=np.asfortranarray(y[:, i]),
          sr=self.sr,
          n_bins=self.n_bins,
          hop_length=self.hop_length,
          fmin=self.fmin,
          bins_per_octave=self.bins_per_octave,
          tuning=self.tuning,
          filter_scale=self.filter_scale,
          norm=self.norm,
          sparsity=self.sparsity,
          window=self.window,
          scale=self.scale,
          pad_mode=self.pad_mode)
      # cqt变换得到复数(complex)
      # 通过abs函数求复数与共轭复数乘积的平方根
      yi = np.abs(xi)[:, :, np.newaxis].astype(self.np_dtype)
      outputs.append(yi)
    return np.concatenate(outputs, axis=2)
  @tf.function
  def tf_cqt(self, y):
    return tf.numpy_function(self.np_cqt, [y], self.tf_dtype)
  def call(self, inputs):
    outputs = inputs
    input_shape = tf.keras.backend.int_shape(inputs)
    assert len(input_shape) in [2, 3], f"The legal nD of input " \
        f"shape of `ToCqt` must be 2 or 3, but got {len(input_shape)}"
    if len(input_shape) == 2:
      outputs = self.tf_cqt(outputs)
    if len(input_shape) == 3:
      temp = []
      for i in range(input_shape[0]):
        xi = tf.keras.layers.Lambda(
            lambda x, t: x[t, :, :],
            output_shape=input_shape[1:],
            arguments={'t': i})(outputs)
        yi = self.tf_cqt(xi)
        temp.append(tf.expand_dims(yi, 0))
      outputs = tf.keras.layers.Concatenate(axis=0)(temp)
    return outputs


if __name__ == "__main__":
  filepath = './dataset/york/1_Vox.wav'
  y0, sr = lrs.load(filepath, sr=None, mono=False)
  y0 = y0.transpose(1, 0)
  y1 = tf.Variable([y0, y0])
  print(y1.shape)
  y2 = ToStft()(y1)
  print(y2.shape)
  y3 = ToMel(sr)(y1)
  print(y3.shape)
  y4 = ToCqt('cqt', sr)(y1)
  print(y4.shape)

