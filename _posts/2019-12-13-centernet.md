---
layout: post
title: CenterNet
date: 2019-12-13
Author: Katherinaxxx
tags: [object detection]
excerpt: " "
image: "/images/post/centernet/ .jpg"
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

近年来，Object Detection 方向的很多工作已经针对检测器在准确性和速度之间的权衡进行了广泛的研究。由于许多 single-stage detector 在两者中都取得了不错的效果，two-stage 的方法逐渐失去了优势。本文首先对 single-stage 和 two-stage 的方法进行简要回顾和比较，之后介绍 anchor-free 的方法并针对以两种 CenterNet 为主的物体检测方法进行详述。

### code

[论文源码](https://github.com/xingyizhou/CenterNet)
