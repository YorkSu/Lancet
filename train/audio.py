# -*- coding: utf-8 -*-
"""Audio

  File: 
    /Lancet/train/audio

  Description: 
    音频处理函数
    使用Tensorflow运算
"""


import tensorflow as tf


def stft(
    Tensor,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    window='hann',
    center=True,
    pad_mode='reflect',
    dtype=tf.complex64):
  """Short-time Fourier transform (STFT)

    Description:
      The STFT represents a signal in the time-frequency domain 
      by computing discrete Fourier transforms (DFT) over short 
      overlapping windows.
  
    Args:
      FIXME

    Return:
      Tensor[shape=(1+n_fft/2, n_frames), dtype=dtype]
  """


if __name__ == "__main__":
  y = tf.random.normal(dtype=tf.float32, shape=[441000])

