# -*- coding: utf-8 -*-
"""UNet

  File: 
    /Lancet/train/model/unet

  Description: 
    UNet模型
"""


from Lancet.core.model import nn
from Lancet.core.model import network


class UNet(network.Network):
  """UNet

    Description: 
      UNet
      FIXME
  """
  def args(self):
    self.filters = [16 * (2 ** i) for i in range(6)]
    self.kernel_shape = 5
    self.strides = 2
    self.activation = 'relu'
    self.output_activation = 'relu'
    self.drop = 0.5

  def build(self):
    # define get layer function
    def conv(filters):
      return nn.convbn(
          filters,
          self.kernel_shape,
          self.strides,
          activation=self.activation)
    def convt(filters):
      return nn.convbn(
          filters,
          self.kernel_shape,
          self.strides,
          activation=self.activation,
          order=3,
          transpose=True)
    def merge():
      def _inner(Tensor1, Tensor2):
        shape1 = nn.get_shape(Tensor1)[1:3]
        Tensor2 = nn.resolutionscal2d(shape1)(Tensor2)
        return nn.concat()([Tensor1, Tensor2])
      return _inner
    # input shape is (None, 1025, n, 2x2)
    inputs = nn.input(self.input_shape)
    x = inputs
    # Conv Part
    x1 = conv(self.filters[0])(x)
    x2 = conv(self.filters[1])(x1)
    x3 = conv(self.filters[2])(x2)
    x4 = conv(self.filters[3])(x3)
    x5 = conv(self.filters[4])(x4)
    x6 = conv(self.filters[5])(x5)
    # DeConv Part
    xt1 = convt(self.filters[4])(x6)
    xt1 = nn.dropout(self.drop)(xt1)
    xt1 = merge()(x5, xt1)
    xt2 = convt(self.filters[3])(xt1)
    xt2 = nn.dropout(self.drop)(xt2)
    xt2 = merge()(x4, xt2)
    xt3 = convt(self.filters[2])(xt2)
    xt3 = nn.dropout(self.drop)(xt3)
    xt3 = merge()(x3, xt3)
    xt4 = convt(self.filters[1])(xt3)
    xt4 = merge()(x2, xt4)
    xt5 = convt(self.filters[0])(xt4)
    xt5 = merge()(x1, xt5)
    # DeConv Part Last Layer
    xt6 = convt(1)(xt5)
    xt6 = nn.resolutionscal2d(self.input_shape[:2])(xt6)
    x = xt6
    outputs = nn.conv2d(
        self.input_shape[-1],
        (4, 4),
        activation=self.output_activation)(xt6)
    outputs = nn.layers.Multiply()([outputs, inputs])
    return nn.model(inputs, outputs)


def unet(*args, **kwargs):
  """unet"""
  return UNet(*args, **kwargs)


if __name__ == "__main__":
  model = unet((1025, 512, 4), (1025, 512, 4))
  model.summary()
  
