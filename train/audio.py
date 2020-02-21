# -*- coding: utf-8 -*-
"""Audio

  File: 
    /Lancet/train/audio

  Description: 
    音频处理函数
    使用Tensorflow运算
"""


import math

import tensorflow as tf


def de_complex(Tensor):
  """Transform tf.complex to 2 tf.float
  
    Description:
      FIXME

    Args:
      FIXME

    Return:
      Tensor[shape=(..., channel*2), dtype=tf.float]
  """
  Tensor = tf.convert_to_tensor(Tensor)
  assert Tensor.dtype in [tf.complex64, tf.complex128], \
      f"[Invalid] Tensor.dtype must be tf.complex, get{type(Tensor)}"
  t_shape = list(Tensor.get_shape())
  outputs = []
  for i in range(t_shape[-1]):
    outputs.extend([
        tf.math.real(Tensor[..., i]),
        tf.math.imag(Tensor[..., i])])
  return tf.stack(outputs, axis=-1)


def to_complex(Tensor):
  """Transform to tf.complex

    Description:
      FIXME

    Args:
      FIXME

    Return:
      Tensor[shape=(..., channel/2), dtype=tf.complex]
  """
  Tensor = tf.convert_to_tensor(Tensor)
  assert Tensor.dtype in [tf.float32, tf.float64], \
      f"[Invalid] Tensor.dtype must be tf.float, get{type(Tensor)}"
  t_shape = list(Tensor.get_shape())
  assert t_shape[-1] % 2 == 0, \
      f"[Invalid] Tensor.channel must be even, get{t_shape[-1]}"
  outputs = []
  for i in range(t_shape[-1] // 2):
    outputs.append(tf.complex(
        Tensor[..., i* 2],
        Tensor[..., i * 2 + 1]))
  return tf.stack(outputs, axis=-1)


def split(Tensor):
  """Split Channel

    Description:
      In `Lancet`, Wave Signal Tensor's Channel is the last axis.
      FIXME

    Args:
      FIXME

    Return:
      List of Tensor[without channel axis]
  """
  Tensor = tf.convert_to_tensor(Tensor)
  # t_shape = list(Tensor.get_shape())
  # t_shape = Tensor.shape
  
  outputs = []
  for i in tf.range(Tensor.shape[-1]):
    outputs.append(Tensor[..., i])
  return outputs


def merge(Tensors):
  """Merge Channel

    Description:
      In `Lancet`, Wave Signal Tensor's Channel is the last axis.
      FIXME

    Args:
      Tensors: List of Tensor, Tensor.shape must be consistent.
      FIXME

    Return:
      Tensor[with channel axis]
  """
  outputs = []
  for Tensor in Tensors:
    outputs.append(tf.convert_to_tensor(Tensor))
  return tf.stack(outputs, axis=-1)


def auto_channel(func):
  """Auto Split and Merge Channel

    Description:
      A Python.Decorator for stft/istft
      if Args.channel is True, this function automatically split
      and merge the channel.
  """
  def inner_func(*args, channel=True, **kwargs):
    if channel:
      if args:
        Tensor = args[0]
      else:
        Tensor = kwargs.pop('Tensor')
      return merge([func(t, *args[1:], **kwargs) for t in split(Tensor)])
    return func(*args, **kwargs)
  return inner_func

@auto_channel
def swap(Tensor, **kwargs):
  """Swap

    Description:
      Swap the Tensor's spectrogram axis
  
    Args:
      Tensor: tf.Tensor[shape=(..., n, n_frames, ...)]
      FIXME

    Return:
      Tensor[shape=(..., n_frames, n, ...)]
  """
  del kwargs
  Tensor = tf.convert_to_tensor(Tensor)
  # t_shape = list(Tensor.get_shape())
  rank = len(Tensor.get_shape())
  axis = list(range(rank - 2)) + [rank - 1, rank - 2]
  return tf.transpose(Tensor, axis)

@auto_channel
def cutting(Tensor, length, **kwargs):
  """Cutting
    
    Description:
      Cutting the Wave Signal Tensor's length (Sample)
  
    Args:
      Tensor: tf.Tensor[shape=(..., Sample, channel)]
      length: Int. the target Sample length.
      Channel: Bool, default True.

    Return:
      Tensor[shape=(..., length, channel)]
  """
  del kwargs
  Tensor = tf.convert_to_tensor(Tensor)
  t_shape = list(Tensor.get_shape())
  assert t_shape[-1] > length, \
      f"[Invalid] length too large, got {length}, sample {t_shape[-1]}"
  outputs = Tensor[..., :length]
  return outputs

@auto_channel
def padding(Tensor, length, pad_mode='reflect', **kwargs):
  """Padding
    
    Description:
      Padding the Wave Signal Tensor's length (Sample)
  
    Args:
      Tensor: tf.Tensor[shape=(..., Sample, channel)]
      length: Int. the target Sample length.
      Channel: Bool, default True.
      pad_mode: Str, default 'reflect'. See tf.pad.mode

    Return:
      Tensor[shape=(..., length, channel)]
  """
  del kwargs
  Tensor = tf.convert_to_tensor(Tensor)
  t_shape = list(Tensor.get_shape())
  t_rank = len(t_shape)
  assert t_shape[-1] < length, \
      f"[Invalid] length too small, got {length}, sample {t_shape[-1]}"
  pad = [[0, length - t_shape[-1]]]
  if t_rank > 1:
    pad = [[0, 0]] * (t_rank - 1) + pad
  return tf.pad(Tensor, paddings=pad, mode=pad_mode)

@auto_channel
def to_length(Tensor, length, pad_mode='reflect', **kwargs):
  """Convert to target length
  
    Description:
      Convert the Wave Signal Tensor's length to target length (Sample)
  
    Args:
      Tensor: tf.Tensor[shape=(..., Sample, channel)]
      length: Int. the target Sample length.
      Channel: Bool, default True.
      pad_mode: Str, default 'reflect'. See tf.pad.mode

    Return:
      Tensor[shape=(..., length, channel)]
  """
  del kwargs
  Tensor = tf.convert_to_tensor(Tensor)
  t_shape = list(Tensor.get_shape())
  t_rank = len(t_shape)
  if t_shape[-1] > length:
    outputs = Tensor[..., :length]
  elif t_shape[-1] < length:
    pad = [[0, length - t_shape[-1]]]
    if t_rank > 1:
      pad = [[0, 0]] * (t_rank - 1) + pad
    outputs = tf.pad(Tensor, paddings=pad, mode=pad_mode)
  else:
    outputs = Tensor
  return outputs

# @auto_channel
# @tf.function
def stft(
    Tensor,
    fft_length=2048,
    frame_length=None,
    frame_step=None,
    window_fn=tf.signal.hann_window,
    pad_end=True,
    pad_mode='reflect',
    name=None,
    **kwargs):
  """Short-time Fourier transform (STFT)

    Description:
      The STFT represents a signal in the time-frequency domain 
      by computing discrete Fourier transforms (DFT) over short 
      overlapping windows.
  
    Args:
      pad_mode: Str, default 'reflect'. See tf.pad.mode
      FIXME

    Return:
      Tensor[shape=(n_frames, 1+fft_length/2), dtype=tf.complex64]
  """
  del kwargs
  with tf.name_scope(name or 'Lancet_stft'):
    Tensor = tf.convert_to_tensor(Tensor, name='Tensor')
    Tensor.shape.with_rank_at_least(1)
    fft_length = tf.convert_to_tensor(fft_length, name='frame_length')
    fft_length.shape.assert_has_rank(0)
    if frame_length is None:
      frame_length = fft_length
    frame_length = tf.convert_to_tensor(frame_length, name='frame_length')
    frame_length.shape.assert_has_rank(0)
    if frame_step is None:
      frame_step = fft_length // 4
    frame_step = tf.convert_to_tensor(frame_step, name='frame_step')
    frame_step.shape.assert_has_rank(0)
    
    framed_signals = tf.signal.frame(
        Tensor, frame_length, frame_step, pad_end=pad_end)

    if window_fn is not None:
      window = window_fn(frame_length, dtype=framed_signals.dtype)
      framed_signals *= window

    return tf.signal.rfft(framed_signals, [fft_length])
  # fft_length = tf.convert_to_tensor(fft_length, name='fft_length')
  # if frame_length is None:
  #   frame_length = tf.convert_to_tensor(fft_length, name='frame_length')
  # if frame_step is None:
  #   frame_step = tf.convert_to_tensor(frame_length // 4, name='frame_step')
  
  # Tensor = tf.convert_to_tensor(Tensor)
  # t_shape = Tensor.shape # list(Tensor.get_shape())
  # t_rank = tf.rank(Tensor) # len(t_shape)

  # if pad_end:
  #   pad = tf.convert_to_tensor([[0, frame_length - t_shape[-1] % frame_step]], dtype=tf.int32)
  #   if t_rank > 1:
  #     pad = tf.concat([tf.zeros(shape=(t_rank - 1, 2), dtype=tf.int32), pad], axis=0)
  #   Tensor = tf.pad(Tensor, paddings=pad, mode=pad_mode)
  # # f_shape = t_shape[:-1] + [int(frame_length), math.ceil(t_shape[-1] / int(frame_step))]
  
  # frame = []
  # for i in tf.range(tf.math.ceil(t_shape[-1] / frame_step), dtype=tf.int32):# f_shape[-1]
  #   frame.append(Tensor[..., i*frame_step:i*frame_step + frame_length])
  # frame = tf.stack(frame, axis=-1)
  # # frame = tf.transpose(frame, list(range(t_rank - 2)) + [t_rank - 1, t_rank - 2])
  # frame = swap(frame, channel=False)

  # if window_fn is not None:
  #   window = window_fn(frame_length, dtype=frame.dtype)
  #   frame *= window

  # return tf.signal.rfft(frame, [fft_length])

@auto_channel
def istft(
    Tensor,
    fft_length=None,
    frame_length=None,
    frame_step=None,
    length=None,
    window_fn=tf.signal.hann_window,
    **kwargs):
  """Inverse Short-time Fourier transform (ISTFT)

    Description:
      Converts a complex-valued spectrogram `stft_matrix` to time-series `y`
      by minimizing the mean squared error between `stft_matrix` and STFT of
      `y` as described in up to Section 2 (reconstruction from MSTFT).
  
      In general, window function, hop length and other parameters should be 
      same as in stft, which mostly leads to perfect reconstruction of a 
      signal from unmodified `stft_matrix`.

    Args:
      Tensor: tf.Tensor[shape=(n_frames, 1+fft_length/2), 
          dtype=tf.complex64]
      pad_mode: Str, default 'reflect'. See tf.pad.mode
      FIXME

    Return:
      Tensor[shape=(..., n), dtype=tf.float]
  """
  del kwargs
  Tensor = tf.convert_to_tensor(Tensor)
  t_shape = list(Tensor.get_shape())

  if fft_length is None:
    fft_length = 2 * (t_shape[-1] - 1)

  fft_length = tf.convert_to_tensor(fft_length, name='fft_length')
  if frame_length is None:
    frame_length = tf.convert_to_tensor(fft_length, name='frame_length')
  if frame_step is None:
    frame_step = tf.convert_to_tensor(frame_length // 4, name='frame_step')
  
  raw_frame = tf.signal.irfft(Tensor, [fft_length])
  if window_fn is not None:
    window = window_fn(frame_length, dtype=Tensor.dtype.real_dtype)
    raw_frame *= window
  
  real_frame = tf.unstack(raw_frame, axis=-2, name="unstack")
  w_shape = list(real_frame[0].get_shape())
  step_shape = list(w_shape[:-1]) + [int(frame_step)]
  block = tf.zeros(shape=w_shape, dtype=tf.float32)
  step = tf.zeros(shape=step_shape, dtype=tf.float32)
  
  slices = []
  for frame in real_frame:
    block += frame
    slices.append(block[..., :frame_step])
    block = tf.concat([block[..., frame_step:], step], axis=-1)
  slices.append(block[..., :frame_length - frame_step])

  outputs = tf.concat(slices, axis=-1)

  if length is not None:
    outputs = to_length(outputs, length, channel=False)

  return outputs


if __name__ == "__main__":
  y = tf.random.normal(dtype=tf.float32, shape=[1, 441000])
  y2 = tf.random.normal(dtype=tf.float32, shape=[1, 441000, 2])
  
  ### basic
  stft_x = stft(y, channel=True)
  print(stft_x.shape, stft_x.dtype)
  # istft_x = istft(stft_x, length=441000)
  # print(istft_x.shape, istft_x.dtype)
  # swap1 = swap(stft_x, channel=False)
  # print(swap1.shape)
  # stft_x2 = merge([stft_x])
  # print(stft_x2.shape)
  # swap2 = swap(stft_x2)
  # print(swap2.shape)
  
  ### split&merge, complex
  # splitx = split(y2)
  # print(splitx[0].shape, splitx[1].shape)
  # stft_d = []
  # for channelt in splitx:
  #   stft_d.append(stft(channelt))
  # mergex = merge(stft_d)
  # print(mergex.shape, mergex.dtype)
  # dec = de_complex(mergex)
  # print(dec.shape, dec.dtype)
  # toc = to_complex(dec)
  # print(toc.shape, toc.dtype)
  
  ### length part
  # cut1 = cutting(y, 440000, channel=False)
  # print(cut1.shape)
  # cut2 = cutting(y2, 440000, channel=True)
  # print(cut2.shape)
  # pad1 = padding(cut1, 441000, channel=False)
  # print(pad1.shape)
  # pad2 = padding(cut2, 441000, channel=True)
  # print(pad2.shape)
  # tlen1 = to_length(y, 320000, channel=False)
  # print(tlen1.shape)
  # tlen2 = to_length(y2, 500000, channel=True)
  # print(tlen2.shape)

