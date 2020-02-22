# -*- coding: utf-8 -*-
"""Audio

  File: 
    /Lancet/core/ops/audio

  Description: 
    Audio Operations
"""


import math

import numpy as np
import scipy as sp
import librosa

from Lancet.core.ops import array


def stft(
    signal,
    fft_length=2048,
    frame_length=None,
    frame_step=None,
    window='hann',
    pad_end=True,
    pad_mode='reflect',
    **kwargs):
  """Short-time Fourier transform (STFT)

    Description:
      The STFT represents a signal in the time-frequency domain 
      by computing discrete Fourier transforms (DFT) over short 
      overlapping windows.
  
    Args:
      FIXME

    Return:
      FIXME
      Tensor[shape=(n_frames, 1+fft_length/2), dtype=tf.complex64]
  """
  fft_length = int(fft_length)
  if frame_length is None:
    frame_length = fft_length
  else:
    frame_length = int(frame_length)
  if frame_step is None:
    frame_step = fft_length // 4
  else:
    frame_step = int(frame_step)
  del kwargs

  signal = np.asarray(signal, np.float32)
  s_shape = list(signal.shape)
  s_rank = len(s_shape)
  
  fft_window = get_window(window, fft_length, frame_length, s_rank)

  if pad_end:
    padding = [[0, frame_length - s_shape[-1] % frame_step]]
    if s_rank > 1:
      padding = [[0, 0]] * (s_rank - 1) + padding
    signal = np.pad(signal, padding, mode=pad_mode)
  
  f_shape = s_shape[:-1] + [frame_length, math.ceil(s_shape[-1] / frame_step)]
  frames = []
  for i in range(f_shape[-1]):
    frame = signal[..., i * frame_step:i * frame_step + frame_length]
    frames.append(np.fft.rfft(fft_window * frame, axis=-1))
    # frames.append(signal[..., i * frame_step:i * frame_step + frame_length])
  
  # outputs = []
  # for frame in frames:
  #   outputs.append(np.fft.rfft(fft_window * frame, axis=-1))

  return np.stack(frames, axis=-1)


def get_window(window, fft_length, frame_length, rank, axis=-1, mode='constant'):
  """Get Window

    Description:
      FIXME
  """
  if callable(window):
    fft_window = window(frame_length)
  elif isinstance(window, (str, tuple)) or np.isscalar(window):
    fft_window = sp.signal.get_window(window, frame_length, fftbins=True)
  elif isinstance(window, (np.ndarray, list)):
    if len(window) == frame_length:
      fft_window = np.asarray(window)
    raise Exception(f'Window size mismatch: '
        f'{len(window):d} != {frame_length:d}')
  else:
    raise Exception(f'Invalid window specification: {window}')
  
  n = fft_window.shape[axis]
  lpad = int((fft_length - n) // 2)
  assert lpad >= 0, f"Target size `fft_length` must be at least input " \
      f"size ({n}), got {fft_length}"
  lengths = [(0, 0)] * fft_window.ndim
  lengths[axis] = (lpad, int(fft_length - n - lpad))

  outputs = np.pad(fft_window, lengths, mode=mode)

  n_shape = [1] * rank
  n_shape[axis] = outputs.shape[0]

  return outputs.reshape(n_shape)


if __name__ == "__main__":
  a = np.random.normal(size=[1, 1, 441000])
  s = stft(a)
  print(s.shape, s.dtype)
  # w = get_window('hann', 2048, 2048, 2)
  # print(w, w.shape)

