# -*- coding: utf-8 -*-
"""Layer

  File: 
    /Lancet/core/model/layer

  Description: 
    自定义网络层
"""


import tensorflow as tf

from Lancet.core.model import util


class ConvBN(tf.keras.layers.Layer):
  """Conv2D with BN and Activation

    Description:
      FIXME
  
    Args:
      order: Int, default 1.
        1 - Conv -> BN -> Activation
        2 - BN -> Activation -> Conv
        3 - Conv -> Activation -> BN
        4 - Activation -> BN -> Conv
      transpose: Bool, default False. if True, use Conv2DTranspose instead of Conv2D
  """
  def __init__(
      self,
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
      trainable=True, 
      name=None,
      **kwargs):
    super().__init__(
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
        trainable=trainable,
        name=name,
        **kwargs)
    self.filters = int(filters)
    self.kernel_size = util.normalize_tuple(kernel_size, 2)
    self.strides = util.normalize_tuple(strides, 2)
    self.padding = padding
    self.activation = tf.keras.activations.get(activation)
    self.use_bn = use_bn
    self.use_bias = use_bias
    self.order = order
    self.transpose = transpose
    self.data_format = data_format or tf.keras.backend.image_data_format()
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)
    self.beta_initializer = beta_initializer
    self.gamma_initializer = gamma_initializer
    self.moving_mean_initializer = moving_mean_initializer
    self.moving_variance_initializer = moving_variance_initializer
    self.beta_regularizer = beta_regularizer
    self.gamma_regularizer = gamma_regularizer
    self.beta_constraint = beta_constraint
    self.gamma_constraint = gamma_constraint
    self.renorm = renorm
    self.renorm_clipping = renorm_clipping
    self.renorm_momentum = renorm_momentum
    self.fused = fused
    self.virtual_batch_size = virtual_batch_size
    self.adjustment = adjustment
    self.trainable = trainable

  def build(self, input_shape):
    if self.transpose:
      self.conv2d_layer = tf.keras.layers.Conv2DTranspose(
          filters=self.filters,
          kernel_size=self.kernel_size,
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          activation=None,
          use_bias=self.use_bias,
          activity_regularizer=self.activity_regularizer,
          kernel_initializer=self.kernel_initializer,
          kernel_regularizer=self.kernel_regularizer,
          kernel_constraint=self.kernel_constraint,
          bias_initializer=self.bias_initializer,
          bias_regularizer=self.bias_regularizer,
          bias_constraint=self.bias_constraint,
          name='Conv2DTranspose')
    else:
      self.conv2d_layer = tf.keras.layers.Conv2D(
          filters=self.filters,
          kernel_size=self.kernel_size,
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          activation=None,
          use_bias=self.use_bias,
          activity_regularizer=self.activity_regularizer,
          kernel_initializer=self.kernel_initializer,
          kernel_regularizer=self.kernel_regularizer,
          kernel_constraint=self.kernel_constraint,
          bias_initializer=self.bias_initializer,
          bias_regularizer=self.bias_regularizer,
          bias_constraint=self.bias_constraint,
          name='Conv2D')
    
    self.bn_layer = tf.keras.layers.BatchNormalization(
        axis=self.axis,
        momentum=self.momentum,
        epsilon=self.epsilon,
        center=self.center,
        scale=self.scale,
        beta_initializer=self.beta_initializer,
        gamma_initializer=self.gamma_initializer,
        moving_mean_initializer=self.moving_mean_initializer,
        moving_variance_initializer=self.moving_variance_initializer,
        beta_regularizer=self.beta_regularizer,
        gamma_regularizer=self.gamma_regularizer,
        beta_constraint=self.beta_constraint,
        gamma_constraint=self.gamma_constraint,
        renorm=self.renorm,
        renorm_clipping=self.renorm_clipping,
        renorm_momentum=self.renorm_momentum,
        fused=self.fused,
        virtual_batch_size=self.virtual_batch_size,
        adjustment=self.adjustment,
        trainable=self.trainable,
        name='BN')
    self.built = True
    
  def call(self, inputs, **kwargs):
    del kwargs
    x = inputs
    
    if self.order == 1:
      x = self.conv2d_layer(x)
      if self.use_bn:
        x = self.bn_layer(x)
      if self.activation is not None:
        x = self.activation(x)
    elif self.order == 2:
      if self.use_bn:
        x = self.bn_layer(x)
      if self.activation is not None:
        x = self.activation(x)
      x = self.conv2d_layer(x)
    elif self.order == 3:
      x = self.conv2d_layer(x)
      if self.activation is not None:
        x = self.activation(x)
      if self.use_bn:
        x = self.bn_layer(x)
    elif self.order == 4:
      if self.activation is not None:
        x = self.activation(x)
      if self.use_bn:
        x = self.bn_layer(x)
      x = self.conv2d_layer(x)
    else:
      raise Exception(f"[Invalid] ConvBN.order expect [1, 2, 3, 4], "
          "got {self.order}")

    return x

  def get_config(self):
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'activation': tf.keras.activations.serialize(self.activation),
        'use_bn': self.use_bn,
        'use_bias': self.use_bias,
        'order': self.order,
        'transpose': self.transpose,
        'data_format': self.data_format,
        'axis': self.axis,
        'momentum': self.momentum,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'kernel_initializer':
            tf.keras.initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            tf.keras.initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            tf.keras.constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            tf.keras.constraints.serialize(self.bias_constraint),
        'beta_initializer':
            tf.keras.initializers.serialize(self.beta_initializer),
        'gamma_initializer':
            tf.keras.initializers.serialize(self.gamma_initializer),
        'moving_mean_initializer':
            tf.keras.initializers.serialize(self.moving_mean_initializer),
        'moving_variance_initializer':
            tf.keras.initializers.serialize(self.moving_variance_initializer),
        'beta_regularizer':
            tf.keras.regularizers.serialize(self.beta_regularizer),
        'gamma_regularizer':
            tf.keras.regularizers.serialize(self.gamma_regularizer),
        'beta_constraint':
            tf.keras.constraints.serialize(self.beta_constraint),
        'gamma_constraint':
            tf.keras.constraints.serialize(self.gamma_constraint)}
    if self.renorm:
      config['renorm'] = True
      config['renorm_clipping'] = self.renorm_clipping
      config['renorm_momentum'] = self.renorm_momentum
    if self.virtual_batch_size is not None:
      config['virtual_batch_size'] = self.virtual_batch_size
    return dict(list(super().get_config().items()) + list(config.items()))


class ResolutionScal2D(tf.keras.layers.Layer):
  """ResolutionScal2D
  
    Description:
      None
    
    Args:
      size: Int or list of 2 Int.
      data_format: Str, default None. `channels_last`(None) or `channels_first`.

    Returns:
      tf.Tensor

    Raises:
      TypeError
      LenError

    Usage:
      None
  """
  def __init__(
      self,
      size: list,
      data_format=None,
      **kwargs):
    super().__init__(trainable=False, **kwargs)
    self.size = util.normalize_tuple(size, 2)
    self.data_format = data_format or tf.keras.backend.image_data_format()

  def call(self, inputs, **kwargs):
    x = inputs
    
    if self.data_format == 'channel_first':
      _size = tf.keras.backend.int_shape(x)[2:4]
    else:
      _size = tf.keras.backend.int_shape(x)[1:3]
    dh = self.size[0] - _size[0]
    dw = self.size[1] - _size[1]
    nh = abs(dh) // 2
    nw = abs(dw) // 2
    lh = [nh, abs(dh) - nh]
    lw = [nw, abs(dw) - nw]
    
    if dh < 0:
      x = tf.keras.layers.Cropping2D([lh, [0, 0]])(x)
    elif dh > 0:
      x = tf.keras.layers.ZeroPadding2D([lh, [0, 0]])(x)
    if dw < 0:
      x = tf.keras.layers.Cropping2D([[0, 0], lw])(x)
    elif dw > 0:
      x = tf.keras.layers.ZeroPadding2D([[0, 0], lw])(x)

    return x
    
  def get_config(self):
    config = {
        'size': self.size,
        'data_format': self.data_format,}
    return dict(list(super().get_config().items()) + list(config.items()))


class Split(tf.keras.layers.Layer):
  """Split
  
    Description:
      FIXME
  """
  def __init__(self, axis=-1, **kwargs):
    super().__init__(**kwargs)
    self.axis = axis

  def build(self, input_shape):
    self.built = True

  def call(self, inputs):
    return 

  def get_config(self):
    config = {
        'axis': self.axis,}
    return dict(list(super().get_config().items()) + list(config.items()))


if __name__ == "__main__":
  y = tf.random.normal(shape=[1, 32, 32, 3], dtype=tf.float32)
  y1 = ConvBN(16, 3)(y)
  print(y1.shape)

