---
layout: post
title: CenterNet
date: 2019-12-13
Author: Katherinaxxx
tags: [object detection]
excerpt: " "
image: "/images/post/centernet/1/1.jpg"
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

> 在相邻时期有两篇CenterNet，如下。
[CenterNet :Objects as Points]()
[CenterNet: Keypoint Triplets for Object Detection](http://openaccess.thecvf.com/content_ICCV_2019/papers/Duan_CenterNet_Keypoint_Triplets_for_Object_Detection_ICCV_2019_paper.pdf)

## CenterNet

> [论文也撞衫，你更喜欢哪个无锚点CenterNet？](https://www.jiqizhixin.com/articles/2019-09-17-6)

### CenterNet :Objects as Points
> [扔掉anchor！真正的CenterNet——Objects as Points论文解读](https://zhuanlan.zhihu.com/p/66048276)

anchor-free目标检测属于anchor-free系列的目标检测，相比于CornerNet做出了改进，使得检测速度和精度相比于one-stage和two-stage的框架都有不小的提高，尤其是与YOLOv3作比较，在相同速度的条件下，CenterNet的精度比YOLOv3提高了4个左右的点。



#### two-stage方法存在的问题

1. 需要很多目标的位置以及区分，这很浪费且低效

2. 不能end-to-end训练

#### 介绍

将单个点作为目标---其边界框的中心点。检测器使用**关键点估计**来找到中心点，然后回归到所有其他目标属性，例如大小，3D位置，方向和角度。

![f](https://katherinaxxx.github.io/images/post/centernet/1/1.jpg#width-full){:height="90%" width="90%"}

**CenterNet方法与基于锚的一阶段方法密切相关。中心点可以看作是单个不可知形状的锚**。但是，有一些重要的区别。
首先，CenterNet仅根据位置而不是boundingbox重叠来分配“锚点”。 对于前景和背景分类，我们没有人工阈值。
其次，每个对象只有一个正“锚”，因此不需要非最大抑制（NMS）。 仅在关键点热图中提取局部峰。
第三，与传统的目标检测器（输出步幅为16）相比，CenterNet使用更大的输出分辨率（输出步幅为4）。这消除了对多个锚点的需求。

![f](https://katherinaxxx.github.io/images/post/centernet/1/mao.jpg#width-full){:height="90%" width="90%"}

**CenterNet与其他keypoint estimation方法相比不需要分组与后续处理，更简单。**
CenterNet是在keypoint estimation的基础上的改进

![f](https://katherinaxxx.github.io/images/post/centernet/1/lunwen.jpg#width-full){:height="90%" width="90%"}

#### 用到的网络

论文中CenterNet提到了三种用于目标检测的网络，这三种网络都是编码解码(encoder-decoder)的结构：

1. Resnet-18 with up-convolutional layers : 28.1% coco and 142 FPS
2. DLA-34 : 37.4% COCOAP and 52 FPS
3. Hourglass-104 : 45.1% COCOAP and 1.4 FPS

每个网络内部的结构不同，但是在模型的最后都是加了三个网络构造来输出预测值，默认是80个类、2个预测的中心点坐标、2个中心点的偏置。

> 因为上文中对图像进行了R=4的下采样，这样的特征图重新映射到原始图像上的时候会带来精度误差，因此对于每一个中心点，额外采用了一个offset偏置

#### 推断阶段

在预测阶段，首先针对一张图像进行下采样，随后对下采样后的图像进行预测，对于每个类在下采样的特征图中预测中心点，然后将输出图中的每个类的热点单独地提取出来。具体提取就是，检测当前热点的值是否比周围的八个近邻点(八方位)都大(或者等于)，然后取100个这样的点，采用的方式是一个3x3的MaxPool，类似于anchor-based检测中nms的效果。

---

### CenterNet: Keypoint Triplets for Object Detection

在ConerNet基础上的提升

#### 存在的问题

#### CenterNet

* ConerNet triplet
* 两种
1.
2.

##### ConerNet


## code

[论文源码](https://github.com/xingyizhou/CenterNet)
