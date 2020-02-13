---
layout: post
title: Collision-Free Video Synopsis Incorporating Object
Speed and Size Changes
date: 2020-01-15
Author: Katherinaxxx
tags: [video]
excerpt: "（监控）视频摘要、视频压缩、融合 "
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

## 问题提出

监控视频通常都是24小时录制，包含有限的有用的信息，因而为了提取有用的信息需要删除冗余和无用的部分。这就是视频摘要/视频压缩/视频融合。

由于监控视频是静态背景下拍摄的，目标很容易检测和提取，因此压缩监控视频的一个有效方法是移动运动中的目标。但是这就会带来两个问题，即**目标碰撞/重叠和破坏时间连续**。

**本文提出**
1）一种**统一**了目标移动、改变速度、改变尺度这三种操作的视频融合的方法，可以避免上述问题。基本思想是要改变移动的目标。
2）提出用Metropolis采样算法替代交替优化作为优化求解的算法。

## 相关研究

现有的视频摘要方法可以大致分为基于框架的视频摘要和基于目标的视频摘要。

1. 基于框架的视频摘要
将视频视为许多的的blocks，依据一定的标准将他们分为重要的和可跳过的

2. 基于目标的视频摘要
把视频中不同时间出现的目标通过编辑放到一起同时出现。


## 提出的方法

### 总览

fig1左展示了shifting，会产生碰撞；中展示了resizing，会避免碰撞；右展示了changing speed，进一步避免碰撞。

这三者并不是分开执行的，而是一个整体

### 背景估计和提取目标管
