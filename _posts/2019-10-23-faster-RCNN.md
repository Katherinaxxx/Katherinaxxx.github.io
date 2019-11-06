---
layout: post
title: 目标检测---Fast RCNN
date: 2019-10-23
Author: Katherinaxxx
tags: [object detection]
excerpt: "论文、代码"
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

## Faster R—CNN

在Fast R-CNN的基础上引入Region Proposal Network（RPN）

### 论文要点

#### RPN

像Fast R-CNN中的用于区域检测的卷积feature map同样可以用于生成候选区域。RPN则是在此之上增加了额外的卷积层做候选框的回归。

#### 结合RPN和fast R-CNN

提出一种在微调候选区域任务和微调目标检测任务之间做变换的训练方法。这样训练收敛很快并且可得到一个统一的网络，两个任务共享卷积feature map

### 代码

论文附带[代码](https://github.com/rbgirshick/py-faster-rcnn)
参考基于tensorflow的[代码](github.com/endernewton/tf-faster-rcnn)实现，解释重要方法
> 知乎上另有详细[代码解释](https://zhuanlan.zhihu.com/p/32230004)
