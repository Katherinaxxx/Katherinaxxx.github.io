---
layout: post
title: CNN Architecture
date: 2019-11-20
Author: Katherinaxxx
tags: [CNN]
excerpt: "了解这些网络架构的衍生或许可以加深理解"
image: "/images/post/cnn/vgg.jpg"
comments: true
toc: true
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

* any list
{:toc}

> [CNN Architectures, a Deep-dive](https://towardsdatascience.com/cnn-architectures-a-deep-dive-a99441d18049)
[cs231n](http://cs231n.stanford.edu/)

这些网络结构常常被迁移至各种学习任务中，做特征提取或者微调一下用于任务，也是常说的backbone
## AlexNet
最初的CNN architecture，因此重要意义所以放在此处，现在其实并没有用了

## VGGNet (Visual Geometry Group)

**VGG的结构** 其实很简单，就是kernal size 3x3不变filter成倍增长的CNN layer的叠加
以下是VGG的**六种**结构，其中VGG16和VGG19比较常用

![vgg](https://katherinaxxx.github.io/images/post/cnn/vgg.jpg#width-full){:height="90%" width="90%"}

**VGG存在的问题** 就在于网络越深后容易出现梯度消失，不过迁移学习和简单的分类还是很好用的。

```python
# 增加了drop out以减小过拟合
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, DropOut
from keras.layers import Flatten, Activation

custom_vgg = Sequential()
custom_vgg.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = (32, 32, 3)))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPooling2D((2, 2)))

custom_vgg.add(Flatten())
custom_vgg.add(Dense(10, activation = "softmax"))

custom_vgg.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

custom_vgg.summary()


# Output params summary
# =================================================================
# Total params: 307,498
# Trainable params: 307,498
# Non-trainable params: 0
# _________________________________________________________________
```
**用处**：Faster R-CNN特征提取

## GoogleNet
GoogleNet网络的大部分初始收益来源于大量地使用降维。

## ResNet

![resnet](https://katherinaxxx.github.io/images/post/cnn/resnet.jpg#width-full){:height="90%" width="90%"}

2015年ImageNet中ResNet有152层，此前没有这么深的，并且错误率低至了3.57%，ResNet提出的**residual block**的结构使得它能实现这样的深度。此外，训练这样深的网络，在每conv层后加BN可以是训练更快并且降低梯度消失。

ResNet有很多变体。**bottleneck**


* ResNext
ILSVRC 2016 classification task第二名，提出了一个新的词“cardinality”，ResNeXt block里路径的数量即为“cardinality”，所有路径的拓扑结构相同。用更大的“cardinality”而非让网络更深或者更宽，可以得到更小的验证误差

![resnext](https://katherinaxxx.github.io/images/post/cnn/resnext.jpg#width-full){:height="90%" width="90%"}

```python
stride = 1
CHANNEL_AXIS = 3

def res_layer(x ,filters,pooling = False,dropout = 0.0):
    temp = x
    temp = Conv2D(filters,(3,3),strides = stride,padding = "same")(temp)
    temp = BatchNormalization(axis = CHANNEL_AXIS)(temp)
    temp = Activation("relu")(temp)
    temp = Conv2D(filters,(3,3),strides = stride,padding = "same")(temp)

    x = add([temp,Conv2D(filters,(3,3),strides = stride,padding = "same")(x)])
    if pooling:
        x = MaxPooling2D((2,2))(x)
    if dropout != 0.0:
        x = Dropout(dropout)(x)
    x = BatchNormalization(axis = CHANNEL_AXIS)(x)
    x = Activation("relu")(x)
    return x

inp = Input(shape = (32,32,3))
x = inp
x = Conv2D(16,(3,3),strides = stride,padding = "same")(x)
x = BatchNormalization(axis = CHANNEL_AXIS)(x)
x = Activation("relu")(x)
x = res_layer(x,32,dropout = 0.2)
x = res_layer(x,32,dropout = 0.3)
x = res_layer(x,32,dropout = 0.4,pooling = True)
x = res_layer(x,64,dropout = 0.2)
x = res_layer(x,64,dropout = 0.2,pooling = True)
x = res_layer(x,256,dropout = 0.4)
x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(4096,activation = "relu")(x)
x = Dropout(0.23)(x)
x = Dense(100,activation = "softmax")(x)

resnet_model = Model(inp,x,name = "Resnet")
resnet_model.summary()

# output params summary
# =================================================
# Total params: 68,671,236
# Trainable params: 68,669,284
# Non-trainable params: 1,952
# _________________________________________________
```


## Dense Net

![densenet](https://katherinaxxx.github.io/images/post/cnn/densenet.jpg#width-full){:height="90%" width="90%"}

在DenseNet中，对于给定层，将其前面的所有其他层连接起来并作为当前层的输入。通过这种安排，我们可以使用更少的滤波器数，并且由于所有层都直接连接到输出，因此可以最大程度地减少消失的梯度问题，可以从输出中直接为每一层计算梯度。

```python
def dense_layer(x, layer_configs):
  layers = []
  for i in range(2):
    if layer_configs[i]["layer_type"] == "Conv2D":
        layer = Conv2D(layer_configs[i]["filters"], layer_configs[i]["kernel_size"], strides = layer_configs[i]["strides"], padding = layer_configs[i]["padding"], activation = layer_configs[i]["activation"])(x)
    layers.append(layer)
  for n in range(2, len(layer_configs)):
    if layer_configs[n]["layer_type"] == "Conv2D":
      layer = Conv2D(layer_configs[n]["filters"], layer_configs[n]["kernel_size"], strides = layer_configs[n]["strides"], padding = layer_configs[n]["padding"], activation = layer_configs[n]["activation"])(concatenate(layers, axis = 3))
    layers.append(layer)
  return layer

  layer_f8 = [
      {
          "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 8, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      }
  ]

  layer_f16 = [
      {
          "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 16, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      }
  ]

  layer_f32 = [
      {
          "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 32, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      }
  ]

  layer_f64 = [
      {
          "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 64, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      }
  ]

  layer_f128 = [
      {
          "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      },{
          "layer_type" : "Conv2D", "filters" : 128, "kernel_size" : (3, 3), "strides" : 1, "padding" : "same", "activation" : "relu"
      }
  ]


  inp = Input(shape = (32, 32, 3))
x = inp
x = Conv2D(4, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
x = dense_layer(x, layer_f8)
x = Dropout(0.8)(x)

x = BatchNormalization(axis = 3)(x)
x = dense_layer(x, layer_f16)
x = Dropout(0.8)(x)

x = BatchNormalization(axis = 3)(x)
x = dense_layer(x, layer_f32)
x = Dropout(0.8)(x)

x = BatchNormalization(axis = 3)(x)
x = dense_layer(x, layer_f64)
x = Dropout(0.8)(x)

x = BatchNormalization(axis = 3)(x)
x = dense_layer(x, layer_f128)
x = Dropout(0.8)(x)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization(axis = 3)(x)
x = Conv2D(96, (1, 1), activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)

x = MaxPooling2D((2, 2))(x)
x = BatchNormalization(axis = 3)(x)
x = Flatten()(x)

x = Dropout(0.4)(x)
x = Dense(14, activation = "softmax")(x)

dense_net = Model(inp, x)
dense_net.summary()

#Output params summary
#==================================================================================================
#Total params: 2,065,686
#Trainable params: 2,064,806
#Non-trainable params: 880
#_____________________________
  ```

与ResNet相比，DenseNet具有更多的中间连接。此外，我们可以在“密集”层中使用较小的过滤器数量，这对于较小的模型非常有用。


## Inception Net

![inception](https://katherinaxxx.github.io/images/post/cnn/inception.jpg#width-full){:height="90%" width="90%"}

在ResNet中，我们创建了更深的网络。Inception Net的想法是使网络更宽。可以通过并行连接具有不同filter的多个层，然后最终将所有这些并行路径串联起来以传递到下一层来完成此操作。

![inceptionmodule](https://katherinaxxx.github.io/images/post/cnn/inceptionmodule.jpg#width-full){:height="90%" width="90%"}

```python
def inception_layer(x, layer_configs):
  layers = []
  for configs in layer_configs:
    if configs[0]["layer_type"] == "Conv2D":
      layer = Conv2D(configs[0]["filters"], configs[0]["kernel_size"], strides = configs[0]["strides"], padding = configs[0]["padding"], activation = configs[0]["activation"])(x)
    if configs[0]["layer_type"] == "MaxPooling2D":
      layer = MaxPooling2D(configs[0]["kernel_size"], strides = configs[0]["strides"], padding = configs[0]["padding"])(x)
    for n in range(1, len(configs)):
      if configs[n]["layer_type"] == "Conv2D":
        layer = Conv2D(configs[n]["filters"], configs[n]["kernel_size"], strides = configs[n]["strides"], padding = configs[n]["padding"], activation = configs[n]["activation"])(layer)
      if configs[n]["layer_type"] == "MaxPooling2D":

        layer = MaxPooling2D(configs[n]["kernel_size"], strides = configs[n]["strides"], padding = configs[n]["padding"])(layer)
    layers.append(layer)
  return concatenate(layers, axis = 3)

  inp = Input(shape = (32, 32, 3))
  x = inp
  x = Conv2D(64, (7, 7), strides = 2, padding = "same", activation = "relu")(x)
  x = MaxPooling2D((3, 3), padding = "same", strides = 2)(x)
  x = Conv2D(64, (1, 1), strides = 1, padding = "same", activation = "relu")(x)
  x = Conv2D(192, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
  x = MaxPooling2D((3, 3), padding = "same", strides = 2)(x)
  x = inception_layer(x, layer_3a)
  x = inception_layer(x, layer_3b)
  x = MaxPooling2D((3, 3), padding = "same", strides = 2)(x)
  x = inception_layer(x, layer_4a)

  x1 = AveragePooling2D((2, 2), strides = 3)(x)
  x1 = Conv2D(128, (1, 1), padding = "same", activation = "relu")(x1)
  x1 = Flatten()(x1)
  x1 = Dense(1024, activation = "relu")(x1)
  x1 = Dropout(0.7)(x1)
  x1 = Dense(100, activation = "softmax")(x1)

  inc = Model(inp, x1)
  inc.summary()

  #Output Params Summary
  #==================================================================================================
  #Total params: 1,373,284
  #Trainable params: 1,373,284
  #Non-trainable params: 0
  #___________________________________________________________

```


**与之前提到的网络结构相比**，InceptionNet是更可取的，因为它们不仅更深，而且更宽，并且我们可以堆叠许多这样的层，但是与所有其他体系结构相比，要训练的输出参数更少。


## InceptionV1
> [Inception系列解读1](https://blog.csdn.net/tominent/article/details/84876337)
[Inception系列解读2](https://blog.csdn.net/yuanchheneducn/article/details/53045551)

**思想**： 加宽；多种不同卷积都用上

**Inception module 的提出主要考虑多个不同 size 的卷积核能够增强网络的适应力**，paper 中分别使用1x1、3x3、5x5卷积核，同时加入3x3 max pooling。

![v1](https://katherinaxxx.github.io/images/post/cnn/v1.jpg#width-full){:height="90%" width="90%"}

如上图所示，每一层 Inception module 的 filters 参数量为所有分支上的总数和，多层 Inception 最终将导致 model 的参数数量庞大，对计算资源有更大的依赖.**解决方法**：1*1卷积层既能跨通道组织信息，提高网络的表达能力，同时可以对输出有效进行降维

___
*还有2个细节：*
1.中间加入2个loss，保证更好的收敛，有正则化作用；
2.最后一个全连接层之前使用的是global average pooling

## InceptionV2

Inception V2 学习了 VGG 用**两个3x3的卷积代替5x5的大卷积**，在降低参数的同时建立了更多的非线性变换，使得 CNN 对特征的学习能力更强

BN 在用于神经网络某层时，会对每一个 mini-batch 数据的内部进行标准化（normalization）处理，使输出规范化到 N(0,1) 的正态分布，减少了 Internal Covariate Shift（内部神经元分布的改变）。

BN 的论文指出，传统的深度神经网络在训练时，每一层的输入的分布都在变化，导致训练变得困难，我们只能使用一个很小的学习速率解决这个问题。而对每一层**使用BN**之后，我们就可以有效地解决这个问题，学习速率可以增大很多倍，达到之前的准确率所需要的迭代次数只有1/14，训练时间大大缩短。而达到之前的准确率后，可以继续训练，并最终取得远超于 Inception V1 模型的性能—— top-5 错误率 4.8%。


## InceptionV3

>[原文](https://arxiv.org/pdf/1512.00567.pdf)
[解析](https://blog.csdn.net/qq_38807688/article/details/84589563)

![v3](https://katherinaxxx.github.io/images/post/cnn/v3.jpg#width-full){:height="90%" width="90%"}

(1)7*7的卷积层替换成了3个3*3的卷积
(2)第一块Inception变成了3个(同BN-Inception)
(3)第一块Inception是传统的
第二块Inception是 Fig.5. 5*5替换成两个3*3
第三块Inception是Fig.6. 1*n 和n*1
v3一个最重要的改进是**分解（Factorization）** ，将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），这第一个样的好处，既可以加速计算（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性

## InceptionV4

Inception模块结合residual connection,极大加速训练，构建Inception-Resnet的同时设计了更深更优化的InceptionV4

## Xception Net

![xception](https://katherinaxxx.github.io/images/post/cnn/xception.jpg#width-full){:height="90%" width="90%"}

就计算效率而言，Xception Net是InceptionNet的即兴创作。Xception表示极端盗梦。上图中显示的Xception体系结构更像ResNet，而不是InceptionNet。Xception Net优于Inception Net v3。

**Inception Net和Xception Net之间的区别在于**，在Inception Net中，执行常规卷积运算，而在Xception Net中，执行深度可分离卷积运算。深度可分离卷积与常规卷积的不同之处在于，在常规Conv2D层中，对于（32、32、3）图像的输入，我们可以在Conv层中使用任意数量的滤镜。这些滤波器中的每一个都将在所有三个通道上运行，并且输出是所有对应值的总和。但是在深度可分离卷积中，每个通道只有一个内核可以进行卷积。因此，通过执行深度可分离卷积，我们可以降低计算复杂度，因为每个内核都是二维的，并且仅在一个通道上进行卷积。

```python
inp = Input(shape = (32, 32, 3))
x = inp
x = Conv2D(32, (3, 3), strides = 2, padding = "same", activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x = Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)

x1 = DepthwiseConv2D((3, 3), (1, 1), padding = "same", activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x1 = DepthwiseConv2D((3, 3), (1, 1), padding = "same", activation = "relu")(x1)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x1 = MaxPooling2D((2, 2), strides = 1)(x1)

x = concatenate([x1, Conv2D(64, (2, 2), strides = 1)(x)])

x1 = Activation("relu")(x)
x1 = Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu")(x1)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x1 = DepthwiseConv2D((3, 3), strides = 1, padding = "same", activation = "relu")(x1)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x1 = DepthwiseConv2D((3, 3), strides = 1, padding = "same")(x1)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x1 = MaxPooling2D((2, 2), strides = 1)(x1)

x = concatenate([x1, Conv2D(256, (2, 2), strides = 1)(x)])


x = Activation("relu")(x)
x = Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x = Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu")(x)
x = BatchNormalization(axis = 3)(x)
x = Dropout(0.4)(x)
x = Flatten()(x)

x = Dense(100, activation = "softmax")(x)


xception = Model(inp, x)
xception.summary()

#Output Params Summary
#==================================================================================================
#Total params: 4,456,548
#Trainable params: 4,454,564
#Non-trainable params: 1,984
#___________________________________________________________________
```

## EfficientNet

2019年发表[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

与其他网络的对比如下

![efficientnet](https://katherinaxxx.github.io/images/post/cnn/efficientnet.jpg#width-full){:height="90%" width="90%"}

用处：EfficientDet的backbone



## 总揽

|  网络  |  特点（改良）  |  优点  |  缺点  |
|  ----  | ----  |  ----  | ----  |
| AlexNet |
| ResNet |
| VGG |
| GoogleNet |
| DenseNet |
| InceptionV1| 1x1，3x3，5x5的conv和3x3的pooling，stack在一起 | 更宽；增加了对尺度的适应性 |
| InceptionV2| 用两个3x3的卷积代替5x5的大卷积；BN| 降低参数量；加速计算 |
| InceptionV3| 将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1）| 加深网络、非线性性；加速计算 |
| InceptionV4| Inception模块结合residual connection | 加速训练 |
| EfficientNet |
