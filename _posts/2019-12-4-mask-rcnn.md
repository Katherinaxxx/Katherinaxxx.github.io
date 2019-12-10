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


## Mask R—CNN

一句话概括Mask R-CNN：Mask RCNN沿用了Faster RCNN的思想，特征提取采用ResNet-FPN的架构，另外多加了一个Mask预测分支

###  Faster R-CNN

详细看这篇[blog](https://katherinaxxx.github.io/blog/faster-RCNN/)

### ResNet-FPN

[FPN](https://arxiv.org/abs/1612.03144)(feature pyramid network)是一种多尺度检测方法。FPN结构中包括自下而上，自上而下和横向连接三个部分。

![fpn](https://katherinaxxx.github.io/images/post/maskrcnn/fpn.jpg#width-full){:height="90%" width="90%"}

这种结构可以将各个层级的特征进行融合，使其同时具有强语义信息和强空间信息，其实可以是特征提取的通用架构，可以和各种bone net相结合。Mask R-CNN则是用的ResNet-FPN，ResNet还可以是：ResNet-50,ResNet-101,ResNeXt-50,ResNeXt-101

### ResNet-FPN + Faster R-CNN + mask

ResNet-FPN + Faster R-CNN + mask其实就是Mask R-CNN。
ResNet-FPN用于特征提取，RPN得到proposals，经过ROIAlign后，一边做目标的分类和框的回归，一边增加卷积层做mask的预测。


## 代码

基于tensorflow和keras的[代码](https://github.com/matterport/Mask_RCNN)
