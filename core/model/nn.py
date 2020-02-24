# -*- coding: utf-8 -*-
"""NN

  File: 
    /Lancet/core/model/nn

  Description: 
    Neural Network Redefine Layers
"""


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers  # pylint:disable=import-error, unused-import

from Lancet.core.model import layer


class Counter(object):
  """Counter 
    
    Description: 
      全局计数器，根据名字自动计数

    Args:
      name: Str. 计数器名字

    Return:
      Str number.

    Usage:
      The `Counter` can be used as a `str` object.
      Or like a function. 
      This two method use the `default initial value 1`.
      And each visit will automatically increase by 1.
  """
  count = {}

  def __init__(self, name=None):
    self.name = name

  def __str__(self):
    if self.name:
      if self.name not in Counter.count:
        Counter.count[self.name] = 1
      else:
        Counter.count[self.name] += 1
      return str(Counter.count[self.name])
    return None


def get_name(name: str, tag=None):
  if tag is None:
    tag = name
    if tag == '':
      addition = tag
    else:
      addition = f"_{Counter(tag)}"
  return name.capitalize() + addition


# ========================
# Layer
# ========================


def reshape(
    target_shape, 
    name=None,
    **kwargs):
  """Reshape Layer"""
  if name is None:
    name = get_name('reshape')
  return tf.keras.layers.Reshape(
      target_shape=target_shape,
      name=name,
      **kwargs)


def flatten(
    data_format=None, 
    name=None,
    **kwargs):
  """Flatten Layer"""
  if name is None:
    name = get_name('flatten')
  return tf.keras.layers.Flatten(
      data_format=data_format,
      name=name,
      **kwargs)


def add(
    name=None,
    **kwargs):
  """Add Layer

    Input must be a list
  """
  if name is None:
    name = get_name('add')
  return tf.keras.layers.Add(
      name=name,
      **kwargs)


def multiply(
    name=None,
    **kwargs):
  """Multiply Layer

    Input must be a list
  """
  if name is None:
    name = get_name('multiply')
  return tf.keras.layers.Multiply(
      name=name,
      **kwargs)


def concatenate(
    axis=-1,
    name=None,
    **kwargs):
  """Concatenate Layer

    input must be a list
  """
  if name is None:
    name = get_name('concatenate')
  return tf.keras.layers.Concatenate(
      axis=axis,
      name=name,
      **kwargs)


def dense( 
    units, 
    activation='relu',
    use_bias=True, 
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', 
    kernel_regularizer=None, 
    bias_regularizer=None,
    activity_regularizer=None, 
    kernel_constraint=None, 
    bias_constraint=None, 
    name=None,
    **kwargs):
  """Full Connect Layer"""
  if activation == 'softmax':
    name = get_name('softmax', '') # 'Softmax'
  elif name is None:
    name = get_name('dense')
  return tf.keras.layers.Dense(
      units=units,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs)


def dropout(
    rate,
    noise_shape=None,
    seed=None,
    name=None,
    **kwargs):
  """Dropout Layer"""
  if name is None:
    name = get_name('dropout')
  return tf.keras.layers.Dropout(
      rate=rate,
      noise_shape=noise_shape,
      seed=seed,
      name=name,
      **kwargs)


def maxpool2d(
    pool_size=(2, 2), 
    strides=None, 
    padding='same',
    data_format=None,
    name=None,
    **kwargs):
  """Max Pooling 2D Layer"""
  if name is None:
    name = get_name('Maxpool2D')
  return tf.keras.layers.MaxPool2D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      name=name,
      **kwargs)


def avgpool2d(
    pool_size=(2, 2), 
    strides=None, 
    padding='same',
    data_format=None,
    name=None,
    **kwargs):
  """Avg Pooling 2D Layer"""
  if name is None:
    name = get_name('Avgpool2D')
  return tf.keras.layers.AvgPool2D(
      pool_size=pool_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      name=name,
      **kwargs)


def globalmaxpool2d(
    data_format=None,
    name=None,
    **kwargs):
  """Global Max Pooling 2D Layer"""
  if name is None:
    name = get_name('GlobalMaxpool2D')
  return tf.keras.layers.GlobalMaxPool2D(
      data_format=data_format,
      name=name,
      **kwargs)


def globalavgpool2d(
    data_format=None,
    name=None, 
    **kwargs):
  """
    Global Avg Pooling 2D Layer
  """
  if name is None:
    name = get_name('GlobalAvgpool2D')
  return tf.keras.layers.GlobalAvgPool2D(
      data_format=data_format,
      name=name,
      **kwargs)


def conv2d(
    filters, 
    kernel_size, 
    strides=1, 
    padding='same',
    data_format=None, 
    dilation_rate=(1, 1), 
    activation=None, 
    use_bias=True,
    kernel_initializer='glorot_uniform', 
    bias_initializer='zeros',
    kernel_regularizer=None, 
    bias_regularizer=None, 
    activity_regularizer=None,
    kernel_constraint=None, 
    bias_constraint=None, 
    name=None,
    **kwargs):
  """Conv2D Layer"""
  if name is None:
    name = get_name('Conv2D')
  return tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs)


def conv3d(
    filters, 
    kernel_size, 
    strides=1, 
    padding='same',
    data_format=None, 
    dilation_rate=(1, 1, 1), 
    activation=None, 
    use_bias=True,
    kernel_initializer='glorot_uniform', 
    bias_initializer='zeros',
    kernel_regularizer=None, 
    bias_regularizer=None, 
    activity_regularizer=None,
    kernel_constraint=None, 
    bias_constraint=None, 
    name=None,
    **kwargs):
  """
    Conv2D Layer
  """
  if name is None:
    name = get_name('Conv3D')
  return tf.keras.layers.Conv3D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      name=name,
      **kwargs)


def dwconv2d(
    kernel_size, 
    strides=1, 
    padding='same',
    depth_multiplier=1,
    data_format=None, 
    activation=None, 
    use_bias=True,
    depthwise_initializer='glorot_uniform', 
    bias_initializer='zeros',
    depthwise_regularizer=None, 
    bias_regularizer=None, 
    activity_regularizer=None,
    depthwise_constraint=None, 
    bias_constraint=None, 
    name=None,
    **kwargs):
  """DWConv2D Layer"""
  if name is None:
    name = get_name('DepthwiseConv2D')
  return tf.keras.layers.DepthwiseConv2D(
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      depth_multiplier=depth_multiplier,
      data_format=data_format,
      activation=activation,
      use_bias=use_bias,
      depthwise_initializer=depthwise_initializer,
      bias_initializer=bias_initializer,
      depthwise_regularizer=depthwise_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      depthwise_constraint=depthwise_constraint,
      bias_constraint=bias_constraint,
      name=name,
      # kernel_initializer=depthwise_initializer, # bug fix
      **kwargs)


def relu(
    max_value=6,
    negative_slope=0,
    threshold=0,
    name=None,
    **kwargs):
  """RuLU Layer

    default:
      max_value: 6
  """
  if name is None:
    name = get_name('relu')
  return tf.keras.layers.ReLU(
      max_value=max_value,
      negative_slope=negative_slope,
      threshold=threshold,
      name=name,
      **kwargs)


def activation(
    activation,
    name=None,
    **kwargs):
  """Activation Layer"""
  if name is None:
    name = get_name('activation')
  return tf.keras.layers.Activation(
      activation=activation,
      name=name,
      **kwargs)


def batchnormalization(
    axis=-1, 
    momentum=0.99, 
    epsilon=1e-3, 
    center=True, 
    scale=True,
    beta_initializer='zeros', 
    gamma_initializer='ones', 
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones', 
    beta_regularizer=None, 
    gamma_regularizer=None,
    beta_constraint=None, 
    gamma_constraint=None, 
    renorm=False, 
    renorm_clipping=None,
    renorm_momentum=0.99, 
    fused=None, 
    trainable=True, 
    virtual_batch_size=None,
    adjustment=None, 
    name=None,
    **kwargs):
  """BatchNormalization Layer"""
  if name is None:
    name = get_name('BatchNormalization')
  return tf.keras.layers.BatchNormalization(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      beta_constraint=beta_constraint,
      gamma_constraint=gamma_constraint,
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      fused=fused,
      trainable=trainable,
      virtual_batch_size=virtual_batch_size,
      adjustment=adjustment,
      name=name,
      **kwargs)


# ========================
# Custom Layer
# ========================


def convbn(
    filters: int,
    kernel_size,
    strides=1,
    padding='same',
    activation=None,
    use_bn=True,
    use_bias=False,
    order=1,
    transpose=False,
    data_format=None,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    axis=-1, 
    momentum=0.99, 
    epsilon=1e-3, 
    center=True, 
    scale=True,
    beta_initializer='zeros', 
    gamma_initializer='ones', 
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones', 
    beta_regularizer=None, 
    gamma_regularizer=None,
    beta_constraint=None, 
    gamma_constraint=None, 
    renorm=False, 
    renorm_clipping=None,
    renorm_momentum=0.99, 
    fused=None, 
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs):
  """Conv2D with BN and Activation
  
    Args:
      order: Int, default 1.
        1 - Conv -> BN -> Activation
        2 - BN -> Activation -> Conv
        3 - Conv -> Activation -> BN
        4 - Activation -> BN -> Conv
      transpose: Bool, default False. if True, use Conv2DTranspose instead of Conv2D
  """
  if name is None:
    name = get_name('ConvBN')
  return layer.ConvBN(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      activation=activation,
      use_bn=use_bn,
      use_bias=use_bias,
      order=order,
      transpose=transpose,
      data_format=data_format,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      beta_constraint=beta_constraint,
      gamma_constraint=gamma_constraint,
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      fused=fused,
      virtual_batch_size=virtual_batch_size,
      adjustment=adjustment,
      name=name,
      **kwargs)


def resolutionscal2d(
    size,
    data_format=None,
    name=None,
    **kwargs):
  """ResolutionScaling2D"""
  if name is None:
    name = get_name('ResolutionScaling2D')
  return layer.ResolutionScal2D(
      size=size,
      data_format=data_format,
      name=name,
      **kwargs)


# ========================
# Other Function
# ========================


def get_shape(Tensor):
  return tf.keras.backend.int_shape(Tensor)


def get_channels(Tensor):
  if tf.keras.backend.image_data_format() == 'channels_first':
    axis = 1
  else:
    axis = -1
  return tf.keras.backend.int_shape(Tensor)[axis]


def set_learning_phase(value):
  tf.keras.backend.set_learning_phase(value)


def repeat(layer, times, *args, **kwargs):
  """Repeat

    Description:
      `repeat` is a `Function` of building Repeat Layers

    Args:
      layer: nn.layer(function)/Keras.layer
      time: int. Number of times that need to be repeated.
      *args & **kwargs: parameters of the nn.layers(function).

    Return: 
      python function

    Usage:
    ```python
      x = nn.repeat(nn.dense, 3, 128)(x)
    ```
  """
  def inner_layer(x):
    for _ in range(times):
      x = layer(*args, **kwargs)(x)
    return x
  return inner_layer


def input(
    shape,
    batch_size=None,
    dtype=None,
    sparse=False,
    tensor=None,
    **kwargs):
  return tf.keras.layers.Input(
      shape=shape,
      batch_size=batch_size,
      dtype=dtype,
      sparse=sparse,
      tensor=tensor,
      name=f"Input",
      **kwargs)


# Alias
model = tf.keras.models.Model
concat = concatenate
local = dense
maxpool = maxpool2d
avgpool = avgpool2d
gmpool = globalmaxpool2d
gapool = globalavgpool2d
conv = conv2d
dwconv = dwconv2d
bn = batchnormalization
# se = sqeuuezeexcitation
# gconv = groupconv2d

