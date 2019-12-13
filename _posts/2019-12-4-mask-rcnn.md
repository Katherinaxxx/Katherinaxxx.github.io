---
layout: post
title: Mask RCNN
date: 2019-12-4
Author: Katherinaxxx
tags: [object detection]
excerpt: "论文、代码"
image: "/images/post/maskrcnn/maskrcnn.jpg"
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

> [Mask R—CNN](https://arxiv.org/abs/1703.06870)
## Mask R—CNN

一句话概括Mask R-CNN：Mask RCNN沿用了Faster RCNN的思想，特征提取采用ResNet-FPN的架构，另外多加了一个对每个RoI的二分Mask预测分支。也就是说，分类、bbox回归和mask分类三者并行。

### mask

用FCN做语义分割的通常做法是用每个像素的softmax和多分类交叉熵损失，这样做masks会涵盖所有的类别。而mask RCNN则使用每个像素的sigmoid和二分损失。实验证明这是取得较好语义分割效果的关键。

**mask representation** 分类和回归都不可避免地通过FCs后缩减成短向量，而mask则是通过像素对像素的卷积操作提取空间结构，这样不会损失空间维度。实验证明，这种fully convolutional representation所需参数更少且更准确。
而这种像素对像素的行为要求RoI的特征与原始保持高度一致，因此提出类RoIAlign layer替代RoI layer，这也是mask预测的关键。

### 网络结构

####  Faster R-CNN

详细看这篇[blog](https://katherinaxxx.github.io/blog/faster-RCNN/)
原始Faster R-CNN的backbone用ResNets从最后的卷积层中提取特征，四个阶段成为C4，用ResNet-50作为backbone记为ResNet-50-C4。

#### ResNet-FPN

寻找其他更有效的backbone，比如就有FPN。
[FPN](https://arxiv.org/abs/1612.03144)(feature pyramid network)是一种多尺度检测方法。FPN结构中包括自下而上，自上而下和横向连接三个部分。

![fpn](https://katherinaxxx.github.io/images/post/maskrcnn/fpn.jpg#width-full){:height="90%" width="90%"}

这种结构可以将各个层级的特征进行融合，使其同时具有强语义信息和强空间信息，其实可以是特征提取的通用架构，可以和各种bone net相结合。Mask R-CNN则是用的ResNet-FPN，ResNet还可以是：ResNet-50,ResNet-101,ResNeXt-50,ResNeXt-101。

ResNet-FPN作为特征提取用在Mask R-CNN中取得了精确度与速度的共同提升。

#### Faster R-CNN + mask

backbone提取完特征后，就是所谓的head net，实际就是Faster R-CNN + mask，具体结构如下所示

![head](https://katherinaxxx.github.io/images/post/maskrcnn/head.jpg#width-full){:height="90%" width="90%"}

这种结构是非常明了的，论文中也提到其实更复杂的设计可能也有巨大潜力能够提升表现，这或许是个可以改进的点。

#### ResNet-FPN + Faster R-CNN + mask

ResNet-FPN + Faster R-CNN + mask其实就是Mask R-CNN。
ResNet-FPN用于特征提取，RPN得到proposals，经过ROIAlign后，一边做目标的分类和框的回归，一边增加卷积层做mask的预测。

### 细节

超参数的设置与Faster R-CNN一致。论文中还讲了其他参数的设置，具体细节还需要查看源码。

## 代码

基于tensorflow和keras的[代码](https://github.com/matterport/Mask_RCNN)
