# -*- coding: utf-8 -*-
"""Layer

  File: 
    /Lancet/train/layer

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


# class Stft(tf.keras.layers.Layer):
#   """Stft
  
#     Description:
#       Based on librosa.stft

#     Args:
#       n_fft: Int, default 2048.
#       hop_length: Int, default None. None -> win_length/4
#       win_length: Int, default None. None -> n_fft
#       window: Str, default 'hann'.
#       center: Bool, default True.
#       pad_mode: Str, default 'reflect'.
#       dtype: Str, default None. None -> 'float32'

#     Input:
#       tf.Tensor, shape=(None, Sample, Channel)

#     Return:
#       tf.Tensor, shape=(None, floor(1+n_fft/2), ceil(Sample/hop_length), Channel)
#   """
#   def __init__(
#       self,
#       n_fft=2048,
#       hop_length=None,
#       win_length=None,
#       window='hann',
#       center=True,
#       pad_mode='reflect',
#       dtype=None,
#       **kwargs):
#     super().__init__(trainable=False, **kwargs)
#     self.n_fft = n_fft
#     self.hop_length = hop_length
#     self.win_length = win_length
#     self.window = window
#     self.center = center
#     self.pad_mode = pad_mode
#     if dtype is None:
#       self.np_dtype = np.float32
#       self.tf_dtype = tf.float32
#       # FIXME: 换成识别全局dtype字符串然后自动选择
#     else:
#       self.np_dtype = dtype
#       self.tf_dtype = dtype
#     self.dtype_ = dtype

#   @tf.function
#   def tf_stft(self, y):
#     """Transform Numpy to TF, Stft"""
#     def np_stft(y):
#       outputs = []
#       for i in range(y.shape[1]):
#         xi = lrs.stft(
#             y=np.asfortranarray(y[:, i]),
#             n_fft=self.n_fft,
#             hop_length=self.hop_length,
#             win_length=self.win_length,
#             window=self.window,
#             center=self.center,
#             pad_mode=self.pad_mode)
#         # stft变换得到复数(complex)
#         # 通过abs函数求复数与共轭复数乘积的平方根
#         yi = np.abs(xi)[:, :, np.newaxis].astype(self.np_dtype)
#         outputs.append(yi)
#       return np.concatenate(outputs, axis=2)
    
#     return tf.numpy_function(np_stft, [y], self.tf_dtype)
  
#   def call(self, inputs):
#     return tf.keras.layers.Lambda(self.tf_stft)(inputs)

#   def get_config(self):
#     config = {
#         'n_fft': self.n_fft,
#         'hop_length': self.hop_length,
#         'win_length': self.win_length,
#         'window': self.window,
#         'center': self.center,
#         'pad_mode': self.pad_mode,
#         'dtype': self.dtype_,}
#     return dict(list(super().get_config().items()) + list(config.items()))


# class Istft(tf.keras.layers.Layer):
#   """Istft
  
#     Description:
#       Based on librosa.istft

#     Args:
#       length: Int, default None. If provided, the output `y` is zero-padded 
#           or clipped to exactly.
#       hop_length: Int, default None. None -> win_length/4
#       win_length: Int, default None. None -> n_fft
#       window: Str, default 'hann'.
#       center: Bool, default True.
#       pad_mode: Str, default 'reflect'.
#       dtype: Str, default None. None -> 'float32'

#     Input:
#       tf.Tensor, shape=(None, floor(1+n_fft/2), ceil(Sample/hop_length), Channel)
      
#     Return:
#       tf.Tensor, shape=(None, length or Sample, Channel)
#   """
#   def __init__(
#       self,
#       length=None,
#       hop_length=None,
#       win_length=None,
#       window='hann',
#       center=True,
#       dtype=None,
#       **kwargs):
#     super().__init__(trainable=False, **kwargs)
#     self.length = length
#     self.hop_length = hop_length
#     self.win_length = win_length
#     self.window = window
#     self.center = center
#     if dtype is None:
#       self.np_dtype = np.float32
#       self.tf_dtype = tf.float32
#       # FIXME: 换成识别全局dtype字符串然后自动选择
#     else:
#       self.np_dtype = dtype
#       self.tf_dtype = dtype
#     self.dtype_ = dtype
    
#   def np_istft(self, y):
#     """Numpy istft"""
#     outputs = []
#     for i in range(y.shape[2]):
#       xi = lrs.istft(
#           stft_matrix=np.asfortranarray(y[:, :, i]),
#           hop_length=self.hop_length,
#           win_length=self.win_length,
#           window=self.window,
#           center=self.center,
#           length=self.length)
#       yi = xi[:, np.newaxis].astype(self.np_dtype)
#       outputs.append(yi)
#     return np.concatenate(outputs, axis=1)
  
#   @tf.function
#   def tf_istft(self, y):
#     """TF istft"""
#     return tf.numpy_function(self.np_istft, [y], self.tf_dtype)
  
#   def call(self, inputs):
#     outputs = inputs
#     input_shape = tf.keras.backend.int_shape(inputs)
#     assert len(input_shape) in [3, 4], f"The legal nD of input " \
#         f"shape of `Istft` must be 3 or 4, but got {len(input_shape)}"
#     if len(input_shape) == 3:
#       outputs = self.tf_istft(outputs)
#     if len(input_shape) == 4:
#       temp = []
#       for i in range(input_shape[0]):
#         xi = tf.keras.layers.Lambda(
#             lambda x, t: x[t, :, :, :],
#             output_shape=input_shape[1:],
#             arguments={'t': i})(outputs)
#         yi = self.tf_istft(xi)
#         temp.append(tf.expand_dims(yi, 0))
#       outputs = tf.keras.layers.Concatenate(axis=0)(temp)
#     return outputs

#   def get_config(self):
#     config = {
#         'length': self.length,
#         'hop_length': self.hop_length,
#         'win_length': self.win_length,
#         'window': self.window,
#         'center': self.center,
#         'dtype': self.dtype_,}
#     return dict(list(super().get_config().items()) + list(config.items()))



class Stft(tf.keras.layers.Layer):
  """Stft
  
    Description:
      Based on tf.signal.stft

    Args:
      n_fft: Int, default 2048.
      hop_length: Int, default None. None -> win_length/4
      win_length: Int, default None. None -> n_fft
      window: Str, default 'hann'.
      center: Bool, default True.
      pad_mode: Str, default 'reflect'.
      dtype: Str, default None. None -> 'float32'

    Input:
      tf.Tensor, shape=(None, Sample, Channel)

    Return:
      tf.Tensor, shape=(None, floor(1+n_fft/2), ceil(Sample/hop_length), Channel)
  """
  def __init__(
      self,
      frame_length=2048,
      frame_step=512,
      fft_length=None,
      window_fn='hann',
      pad_end=True,
      **kwargs):
    super().__init__(trainable=False, **kwargs)
    self.frame_length = frame_length
    self.frame_step = frame_step
    self.fft_length = fft_length
    self.window_fn = window_fn
    self.pad_end = pad_end

  def call(self, inputs):
    input_shape = tf.keras.backend.int_shape(inputs)
    if self.window_fn == 'hann':
      window_fn = tf.signal.hann_window
    else:
      window_fn = self.window_fn
    outputs = []
    if len(input_shape) == 2:
      for i in range(input_shape[-1]):
        xi = tf.keras.layers.Lambda(
            lambda x, t: x[:, t],
            arguments={'t': i})(inputs)
        yi = tf.signal.stft(
            xi,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            window_fn=window_fn,
            pad_end=self.pad_end)
        yi = tf.transpose(yi, [1, 0])
        outputs.append(tf.expand_dims(yi, -1))
    elif len(input_shape) == 3:
      for i in range(input_shape[-1]):
        xi = tf.keras.layers.Lambda(
            lambda x, t: x[:, :, t],
            arguments={'t': i})(inputs)
        yi = tf.signal.stft(
            xi,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            window_fn=window_fn,
            pad_end=self.pad_end)
        yi = tf.transpose(yi, [0, 2, 1])
        outputs.append(tf.expand_dims(yi, -1))
    else:
      raise ValueError(f"[Invalid] nD, got {len(input_shape)}")
    
    # 拆出复数的实部和虚部
    outputs = [
        x
        for c in outputs
        for x in [tf.math.real(c), tf.math.imag(c)]]

    return tf.keras.layers.Concatenate(axis=-1)(outputs)

  def get_config(self):
    config = {
        'frame_length': self.frame_length,
        'frame_step': self.frame_step,
        'fft_length': self.fft_length,
        'window_fn': self.window_fn,
        'pad_end': self.pad_end,}
    return dict(list(super().get_config().items()) + list(config.items()))


class Istft(tf.keras.layers.Layer):
  """Istft
  
    Description:
      Based on tf.signal.inverse_stft

    Args:
      length: Int, default None. If provided, the output `y` is zero-padded 
          or clipped to exactly.
      hop_length: Int, default None. None -> win_length/4
      win_length: Int, default None. None -> n_fft
      window: Str, default 'hann'.
      center: Bool, default True.
      pad_mode: Str, default 'reflect'.
      dtype: Str, default None. None -> 'float32'

    Input:
      tf.Tensor, shape=(None, floor(1+n_fft/2), ceil(Sample/hop_length), Channel)
      
    Return:
      tf.Tensor, shape=(None, length or Sample, Channel)
  """
  def __init__(
      self,
      frame_length=2048,
      frame_step=512,
      fft_length=None,
      window_fn='hann',
      **kwargs):
    super().__init__(trainable=False, **kwargs)
    self.frame_length = frame_length
    self.frame_step = frame_step
    self.fft_length = fft_length
    self.window_fn = window_fn
    
  def call(self, inputs):
    input_shape = tf.keras.backend.int_shape(inputs)
    if self.window_fn == 'hann':
      window_fn = tf.signal.inverse_stft_window_fn(
          self.frame_step,
          forward_window_fn=tf.signal.hann_window)
    else:
      window_fn = self.window_fn
    outputs = []
    if len(input_shape) == 3:
      for i in range(input_shape[-1] // 2):
        xi = tf.keras.layers.Lambda(
            lambda x, t: x[:, :, 2 * t],
            arguments={'t': i})(inputs)
        xj = tf.keras.layers.Lambda(
            lambda x, t: x[:, :, 2 * t + 1],
            arguments={'t': i})(inputs)
        x = tf.complex(xi, xj)
        x = tf.transpose(x, [1, 0])
        y = tf.signal.inverse_stft(
            x,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            window_fn=window_fn)
        outputs.append(tf.expand_dims(y, -1))
    elif len(input_shape) == 4:
      for i in range(input_shape[-1] // 2):
        xi = tf.keras.layers.Lambda(
            lambda x, t: x[:, :, :, 2 * t],
            arguments={'t': i})(inputs)
        xj = tf.keras.layers.Lambda(
            lambda x, t: x[:, :, :, 2 * t + 1],
            arguments={'t': i})(inputs)
        x = tf.complex(xi, xj)
        x = tf.transpose(x, [0, 2, 1])
        y = tf.signal.inverse_stft(
            x,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            window_fn=window_fn)
        outputs.append(tf.expand_dims(y, -1))
    else:
      raise ValueError(f"[Invalid] nD, got {len(input_shape)}")
    
    return tf.keras.layers.Concatenate(axis=-1)(outputs)

  def get_config(self):
    config = {
        'frame_length': self.frame_length,
        'frame_step': self.frame_step,
        'fft_length': self.fft_length,
        'window_fn': self.window_fn,}
    return dict(list(super().get_config().items()) + list(config.items()))


class Mel(tf.keras.layers.Layer):
  """Mel
    
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
      pad_mode: Str, default 'reflect'.
      power: Float, default 2.0.
      dtype: Str, default None. None -> 'float32'

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
      pad_mode='reflect',
      power=2.0,
      dtype=None,
      **kwargs):
    super().__init__(trainable=False, **kwargs)
    self.sr = sr
    self.n_mels = n_mels
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    self.window = window
    self.center = center
    self.pad_mode = pad_mode
    self.power = power
    if dtype is None:
      self.np_dtype = np.float32
      self.tf_dtype = tf.float32
      # FIXME: 换成识别全局dtype字符串然后自动选择
    else:
      self.np_dtype = dtype
      self.tf_dtype = dtype
    self.dtype_ = dtype
    
  def np_mel(self, y):
    """Numpy mel"""
    outputs = []
    for i in range(y.shape[1]):
      xi = lrs.feature.melspectrogram(
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
          power=self.power)
      yi = xi[:, :, np.newaxis].astype(self.np_dtype)
      outputs.append(yi)
    return np.concatenate(outputs, axis=2)
  
  @tf.function
  def tf_mel(self, y):
    """TF mel"""
    return tf.numpy_function(self.np_mel, [y], self.tf_dtype)
  
  def call(self, inputs):
    outputs = inputs
    input_shape = tf.keras.backend.int_shape(inputs)
    assert len(input_shape) in [2, 3], f"The legal nD of input " \
        f"shape of `Mel` must be 2 or 3, but got {len(input_shape)}"
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

  def get_config(self):
    config = {
        'sr': self.sr,
        'n_mels': self.n_mels,
        'n_fft': self.n_fft,
        'hop_length': self.hop_length,
        'win_length': self.win_length,
        'window': self.window,
        'center': self.center,
        'pad_mode': self.pad_mode,
        'power': self.power,
        'dtype': self.dtype_,}
    return dict(list(super().get_config().items()) + list(config.items()))


class Cqt(tf.keras.layers.Layer):
  """Cqt
  
    Description:
      Based on librosa.cqt&hybrid_cqt&pseudo_cqt

    Args:
      mode: Str. 'cqt'->cqt, 'hcqt'&'hybrid_cqt'->hybrid_cqt, 
          'pcqt'&'pseudo_cqt'->pseudo_cqt
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
      tf.Tensor, shape=(None, n_bins, ceil(Sample/hop_length), Channel)
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
    self.dtype_ = dtype
  
  def np_cqt(self, y):
    """Numpy cqt"""
    if self.mode == 'cqt':
      method = lrs.cqt
    elif self.mode in ['hcqt', 'hybrid_cqt']:
      method = lrs.hybrid_cqt
    elif self.mode in ['pcqt', 'pseudo_cqt']:
      method = lrs.pseudo_cqt
    else:
      raise ValueError(f'[Invalid] Cqt.mode, got {self.mode}')
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
    """TF cqt"""
    return tf.numpy_function(self.np_cqt, [y], self.tf_dtype)
  
  def call(self, inputs):
    outputs = inputs
    input_shape = tf.keras.backend.int_shape(inputs)
    assert len(input_shape) in [2, 3], f"The legal nD of input " \
        f"shape of `Cqt` must be 2 or 3, but got {len(input_shape)}"
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

  def get_config(self):
    config = {
        'mode': self.mode,
        'sr': self.sr,
        'n_bins': self.n_bins,
        'hop_length': self.hop_length,
        'fmin': self.fmin,
        'bins_per_octave': self.bins_per_octave,
        'tuning': self.tuning,
        'filter_scale': self.filter_scale,
        'norm': self.norm,
        'sparsity': self.sparsity,
        'window': self.window,
        'scale': self.scale,
        'pad_mode': self.pad_mode,
        'dtype': self.dtype_,}
    return dict(list(super().get_config().items()) + list(config.items()))


if __name__ == "__main__":
  filepath = './dataset/york/1_Vox.wav'
  y0, sr = lrs.load(filepath, sr=None, mono=False)
  y0 = y0.transpose(1, 0)
  y1 = tf.Variable([y0, y0])
  print(y1.shape)
  y2 = Stft()(y1)
  print(y2.shape, y2.dtype)
  # y3 = Mel(sr)(y1)
  # print(y3.shape)
  # y4 = Cqt('cqt', sr)(y1)
  # print(y4.shape)
  # yc = tf.keras.layers.Conv2D(2, 3, padding='same')(y2)
  y5 = Istft()(y2)
  print(y5.shape, y5.dtype)

