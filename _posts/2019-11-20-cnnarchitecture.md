---
layout: post
title: CNN Architecture
date: 2019-11-20
Author: Katherinaxxx
tags: [CNN]
excerpt: "了解这些网络架构的衍生或许可以加深理解"
image: "/images/pic10.jpg"
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
## AlexNet
最初的CNN architecture，因此重要意义所以放在此处，现在其实并没有用了

## VGGNet (Visual Geometry Group)

VGG的结构其实很简单，就是kernal size 3x3不变filter成倍增长的CNN layer的叠加
以下是VGG的六种结构，其中VGG16和VGG19比较常用

![vgg](https://katherinaxxx.github.io/images/post/cnn/vgg.jpg#width-full){:height="90%" width="90%"}

VGG存在的问题就在于网络越深后容易出现梯度消失，不过迁移学习和简单的分类还是很好用的。

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

## GoogleNet


## ResNet

![resnet](https://katherinaxxx.github.io/images/post/cnn/resnet.jpg#width-full){:height="90%" width="90%"}

2015年ImageNet中ResNet有152层，此前没有这么深的，并且错误率低至了3.57%，ResNet提出的residual block的结构使得它能实现这样的深度。此外，训练这样深的网络，在每conv层后加BN可以是训练更快并且降低梯度消失。ResNet有很多变体。

ResNext
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

#output params summary
# =================================================
# Total params: 68,671,236
# Trainable params: 68,669,284
# Non-trainable params: 1,952
# _________________________________________________
```


## Dense Net


## Inception Net


## InceptionV3

>[原文]()
[解析](https://blog.csdn.net/qq_38807688/article/details/84589563)


## Xception Net
