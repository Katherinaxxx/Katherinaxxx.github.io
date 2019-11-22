---
layout: post
title: 目标检测---Fast RCNN
date: 2019-10-23
Author: Katherinaxxx
tags: [object detection]
excerpt: "论文、代码"
image: "/images/post/fasterrcnn/liucheng.jpg"
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

在Fast R-CNN的基础上引入Region Proposal Network（RPN）代替selective search作为选取候选框的方法，大大提速。Faster R-CNN实质上就是RPN+Fast R-CNN

### 论文要点
>[Faster R-CNN理解、讨论](https://blog.csdn.net/shenziheng1/article/details/82907663)

#### RPN

像Fast R-CNN中的用于区域检测的卷积feature map同样可以用于生成候选区域。RPN则是在此之上增加了额外的卷积层做候选框的回归。

**anchor** 是在feature map上滑窗时，滑窗中心在原像素空间的映射点。

> [相关解释](https://blog.csdn.net/gm_margin/article/details/80245470)

negative、positive

回归分类分别基于anchor得到的框计算是否是目标的概率以及与真实框重合的UoI

RPNs干的活儿就是判断anchors属于前景还是背景，再利用BBox regression修正anchors从而获得精确的proposals

##### 训练RPNs

SGD； 256 anchors ；（0，0.01）高斯分布初始化参数；ZF net 、VGGNet； lr=0.0001 decay=0.0005 momentum=0.9


#### 结合RPN和fast R-CNN

提出一种在微调候选区域任务和微调目标检测任务之间做变换的训练方法。这样训练收敛很快并且可得到一个统一的网络，两个任务共享卷积feature map

### 代码

>论文附带[代码](https://github.com/rbgirshick/py-faster-rcnn)是基于caffe的
参考基于tensorflow的[代码](https://github.com/endernewton/tf-faster-rcnn)实现，根据readme运行若不可（不用gpu）[参照](https://blog.csdn.net/m0_38024766/article/details/90712715)，与训练模型下载不下来的可以从[这里](https://drive.google.com/drive/folders/0B1_fAEgxdnvJeGg0LWJZZ1N2aDA)下载
知乎上另有详细[代码解释](https://zhuanlan.zhihu.com/p/32230004)

#### 代码解读

省略掉特征提取后整个流程详细如下

![liucheng](https://katherinaxxx.github.io/images/post/fasterrcnn/liucheng.jpg#width-full){:height="90%" width="90%"}

以下是主要代码结构图

![code](https://katherinaxxx.github.io/images/post/fasterrcnn/code.jpg#width-full){:height="90%" width="90%"}

1. tools/demo.py
展示了用data/demo中图片做目标检测的一个实例



#### demo

mac不支持跑gpu因此按以上参考进行修改，在data下执行
```
xyhdeMacBook-Pro:data xyh$ ../tools/demo.py
```
即可正常运行demo
用的是res101，如果用别的预训练模型

> ps. 执行完demo.py程序后删除tf-faster-rcnn/output/res101/voc_2007_trainval+voc_2012_trainval下的default文件 ，然后再去执行测试文件，因为测试时还要创建一个default文件，那个和现在这个是不一样的，所以如果不删除这个现存的default文件会导致无法测试，显示default文件已存在，创建失败。当然测试完之后如果要运行demo.py文件的话，同样是要删除掉测试时创建的default文件的。

#### 测试模型

#### 训练自己的数据
