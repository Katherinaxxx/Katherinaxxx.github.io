---
layout: post
title: EfficientDet
date: 2019-12-12
Author: Katherinaxxx
tags: [object detection]
excerpt: "很新的论文，记录其中或关键或启发的点"
image: "/images/post/efficientdet/ed.jpg"
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

> [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

## EfficientDet

### 介绍

为了达到现有方法的最佳水平，计算资源消耗巨大，这是现实资源的限制，因此目标检测的效率问题越来越需被重视。本文提出的EfficientDet旨在不牺牲精确度的前提下提升效率。
基于一阶段检测设计，主要存在两个挑战：

1）高效的多尺度特征融合。自从有FPN，诸多方法都在此基础上发展多尺度特征融合的方法，但是绝大多数仅仅不加区分地将不同尺度的特征输入sum，这一操作由于分辨率不同而导致融合不对等。
为了解决，本文提出BiFPN即双向FPN，在FPN的基础上引入权重对不同输入特征加权。

2）模型尺度增加。原先很多方法依靠更大的backbone或者输入图片尺寸来提高精确度，我们观察到增加特征网络和box/class预测网络同样有助于提升精确度和效率。

最终，以EfficientNet作为backbone，结合提出的BiFPN和compound scaling就得到了EfficientDet，这个框架下可以得到更高的精确度但是参数和FLOPS都更少。


![1](https://katherinaxxx.github.io/images/post/efficientdet/1.jpg#width-full){:height="60%" width="60%"}

### BiFPN

BiFPN的结构如下所示：

![bi](https://katherinaxxx.github.io/images/post/efficientdet/bifpn.jpg#width-full){:height="60%" width="60%"}

#### cross-scale connections

自上而下的FPN由于只有单一方向的信息流而受到限制。PANet则增加了自下而上的路径。NAS-FPN则用神经结构搜索去探索更好的交叉尺度网络拓扑结构，但是需要很强的计算能力而且难以复刻。为了提升模型效率，文中也给出了几个优化cross-scale connections的建议：

1）删除只有一个输入的节点，即simplified PANet。直观地想，只有一个输入的对于后面多尺度融合的贡献小。

2）从原始输入到同级的输出增加一条路径，这样就能不增加其他损失的情况下赠融合更多特征。

3）不像PANet只有一个自上而下和自下而上的路径，我们把每个双向路径看作看作一层，且同一层重复多次从而做更多高维特征融合。

#### 加权特征融合

正如上文提到的，大多数直接加和的做法是存在问题的，因此本文增加权重来解决这个问题。然而对于加权融合也有一下几个思路：

1）unbounded fusion。与先前相比有一定精确度的增加，但是由于权重无边界可能导致训练不稳定。因此接下来考虑做标准化处理。

2）softmax-based fusion。用softmax使权重都映射到0-1的概率空间。但实验证明额外的softmax会导致GPU速度显著下降。

3）fast normalized fusion。同样将权重压缩到0-1，但没有使用softmax因此更有效率。后面的消融实验也证明fast normalized fusion有着与softmax-based fusion相似的学习过程和精确度，但是比后者速度快30%。

最终，将cross-scale connections和fast normalized fusion结合起来就得到了BiFPN。为了进一步提高效率，还使用了depthwise separable convolution做特征融合，并在每个卷积后加了BN和激活函数。

### EfficientDet

基于BiFPN，提出了EfficientDet这个新的检测模型。接下来介绍EfficientDet的网络结构和一个新的compound scaling method

#### 网络结构

EfficientDet总的结构如下图所示：

![ed](https://katherinaxxx.github.io/images/post/efficientdet/ed.jpg#width-full){:height="60%" width="60%"}

可以看出基本是符合one-stage检测。用ImageNet与训练的EfficientNet作为backbone，BiFPN作为feature net。backbone的输出 $P_3$ ... $P_7$ 重复的做自上而下和自下而上的双向特征融合。融合后的特征喂给一个class and box net产生目标类别和框。

#### compound scaling

原先很多的做法都是增大backbone（即从深度的角度）这一角度或者有限的角度，因此通常是低效的。最近的研究表明将深度、宽度和输入分辨率这三个角度一起增大会明显提高模型的表现。本文提出的compound scaling就是同时提高backbone、BiFPN、class/box net和分辨率。

网格化搜索太过于耗时耗力，因此基于启发式的增大。具体细节查看论文。

### 实验

通过各种实验表明EfficientDet的优越性以及各个设计或结构的优异表现。

![2](https://katherinaxxx.github.io/images/post/efficientdet/2.jpg#width-full){:height="60%" width="60%"}

![3](https://katherinaxxx.github.io/images/post/efficientdet/3.jpg#width-full){:height="60%" width="60%"}

![4](https://katherinaxxx.github.io/images/post/efficientdet/4.jpg#width-full){:height="60%" width="60%"}

### 总结

总结来说，EfficientDet同时提升了精确度和效率。尤其是可以达到精确度现有技术水平，且参数、FLOPS比现有检测手段更少，在cpu和gpu上跑得也更快。

### code

源码目前还没发出来
