---
layout: post
title: Faster RCNN
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
[一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)
[Object Detection and Classification using R-CNNs](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/)

整个流程如下

![train](https://katherinaxxx.github.io/images/post/fasterrcnn/train.jpg#width-full){:height="90%" width="90%"}

#### 特征提取

feature map 将被用于RPN和FC，比如用VGG（原文）
特点在于，所有的conv层都是：kernel_size=3，pad=1，stride=1
所有的pooling层都是：kernel_size=2，pad=1，stride=1 。因此M*N通过一层（n）conv+relu+pool后变为M/2*N/2，经过四层这样的结构最后都会变成M/16*N/16

>相比较YOLOv3经过下采样分辨率变为原来的1/32，FasterRCNN是1/16，也是因为这个原因FasterRCNN在小物体目标检测上效果优于YOLOv3


#### RPN

RPN的作用是生成候选框。输入为特征提取得到的feature map。

特征提取得到的feature map通过卷积层以外，还增加了bounding box regression来修正从而获得精确的proposals。**因此是两个分支，一个用于计算anchor对应前景背景的概率，一个用于计算anchor的bounding box偏移来定位。**

具体来讲，回归分类分别基于anchor得到的框计算是否是目标的概率以及与真实框重合的UoI（满足以下其一判定为前景：（1）IoU最大[以防（2）无一满足的情况]（2）IoU>0.7），也就是判断anchors属于前景还是背景，再利用BBox regression得到偏移量用来修正anchors从而获得精确的proposals。同时也剔除了过小或超出边界的proposals

**总结RPN：生成anchors -> softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals**

NMS：由于一些RPN proposals高度重合，为了减少重复，基于proposal区域的cls scores和IoU做NMS。可以减少proposal数量但不会影响最终的结果

**anchor** 是在feature map上滑窗时，滑窗中心在原像素空间的映射点。滑窗中心对应k个(k=9)anchors作为初始的检测框，分前景和背景因此clf = 2k scores，同时四个坐标reg = 4k coordinates

> [相关解释](https://blog.csdn.net/gm_margin/article/details/80245470)

**BBox regression**
窗口用(x,y,w,h)表示，其中x、y为中心点坐标w、h则为宽高。目标是寻找一种关系，使得输入原始的anchor A经过映射得到一个跟真实窗口G更接近的回归窗口G'

>对比RPN和YOLOv3
YOLOv3借鉴了RPN。RPN两个分支分开做的，预测前景背景但不会对目标分类；而YOLOv3是整个就是回归得到目标类别和坐标。然后具体的设置细节不一样。
##### 训练RPNs

SGD； 256 anchors ；（0，0.01）高斯分布初始化参数；ZF net 、VGGNet； lr=0.0001 decay=0.0005 momentum=0.9


#### 结合RPN和fast R-CNN

提出一种在微调候选区域任务和微调目标检测任务之间做变换的训练方法。这样训练收敛很快并且可得到一个统一的网络，两个任务共享卷积feature map

### RoI pooling

实质干的是统一尺寸的活儿 有点像SPPNet里的金字塔。
该层输入的是特征提取feature map和RPN给出的目标区域（大小各不相同），综合这些信息后提取目标区域的feature map(相同大小)，送入后续FC判定目标类别。

RoI pooling的改进为roi align （最近邻插值->双线性插值）（取整->保留浮点）
>Mask R-CNN指出RoI pooling存在问题， 即feature map和原始图像不对准从而会影响检测精度，因此提出ROI Align来取代RoI pooling，可以保证大致的位置。
[ROI操作：ROIPooling和ROIAlign的特点和区别](https://baijiahao.baidu.com/s?id=1616632836625777924&wfr=spider&for=pc)

### clf & reg

根据RoI pooling输出的feature map做分别做分类和回归。

### 代码

>论文附带[代码](https://github.com/rbgirshick/py-faster-rcnn)是基于caffe的
参考基于tensorflow的[代码](https://github.com/endernewton/tf-faster-rcnn)实现，根据readme运行若不可（不用gpu）[参照1](https://blog.csdn.net/m0_38024766/article/details/90712715)、[参考2](http://www.jeepxie.net/article/615177.html)。训练模型下载不下来的可以从[这里](https://drive.google.com/drive/folders/0B1_fAEgxdnvJeGg0LWJZZ1N2aDA)下载
知乎上另有详细[代码解释](https://zhuanlan.zhihu.com/p/32230004)

#### 代码解读

省略掉特征提取后整个流程详细如下

![liucheng](https://katherinaxxx.github.io/images/post/fasterrcnn/liucheng.jpg#width-full){:height="90%" width="90%"}

以下是主要代码结构图

![code](https://katherinaxxx.github.io/images/post/fasterrcnn/code.jpg#width-full){:height="90%" width="90%"}

#### demo

1. tools/demo.py
展示了用data/demo中图片做目标检测的一个实例
如果修改了demo中图片应对应修改148行im_names的图片名称

2. 运行
mac不支持跑gpu因此按以上参考进行修改，在data下执行
```
xyhdeMacBook-Pro:data xyh$ ../tools/demo.py
```
即可正常运行demo
用的是res101，如果用别的预训练模型

> ps. 执行完demo.py程序后删除tf-faster-rcnn/output/res101/voc_2007_trainval+voc_2012_trainval下的default文件 ，然后再去执行测试文件，因为测试时还要创建一个default文件，那个和现在这个是不一样的，所以如果不删除这个现存的default文件会导致无法测试，显示default文件已存在，创建失败。当然测试完之后如果要运行demo.py文件的话，同样是要删除掉测试时创建的default文件的。

#### 测试模型

1. experiments/scripts/test_faster_rcnn.sh

2. 运行
基本按之前给的blog操作，但他写的(1)有一点错误，正确为

```
# 软链接
NET=res101
TRAIN_IMDB=voc_2007_trainval
mkdir -p output/${NET}/${TRAIN_IMDB}
cd output/${NET}/${TRAIN_IMDB}
ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default
cd ../../..
```

```
# 下载模型并改名（名字太长会报错）
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzvf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt res101.ckpt
cd ../..
```

```
# 运行
./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
# GPU_ID is the GPU you want to test on
# NET in {vgg16, res50, res101, res152} is the network arch to use
# DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
# Examples:
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
./experiments/scripts/train_faster_rcnn.sh 1 coco res101

./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc res101
```

#### 训练自己的数据
