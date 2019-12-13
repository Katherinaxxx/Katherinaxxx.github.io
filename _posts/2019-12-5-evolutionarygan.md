---
layout: post
title: Evolutionary GAN
date: 2019-12-5
Author: Katherinaxxx
tags: [algorithm]
excerpt: "进化算法、GAN"
image: "/images/post/egan/egan.jpg"
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

## Evolutionary GAN

[Evolutionary GAN](https://ieeexplore.ieee.org/document/8627945)(EGAN)也是一种为了解决GANs不稳定和模式崩困而提出的新的GAN结构。将[进化算法](https://en.wikipedia.org/wiki/Evolutionary_algorithm)引入GAN，因而得名EGAN，使得训练稳定且保证生成多样性。

> [参考](https://www.jianshu.com/p/62b9e0f00305)

与GANs的不同之处在于，GANs用定义好的对抗损失来交替训练D和G，而EGAN是演化出了一堆Gs来和D训练。不同的对抗训练损失作为变异算子，每个G的更新基于这些变异算子。还设计了进化机制来衡量生成样本的质量和多样性，使得表现好的G才能被保留下来继续训练。

### 引言

引言部分介绍GAN、存在的问题以及主要的解决思路存在的问题。最后引出EGAN这种新的结构。
关于主要的解决思路存在的问题，一是改进损失函数陈列了最小二乘、kl散度、Wasserstein距离、测量kl散度，但这些改进也存在问题；二是训练优化算法，固定对抗训练策略后，几乎不能在训练过程中调整D和G的平衡。

EGAN这种新的结构的思路是，将D作为环境（提供适应损失函数），Gs作为种群在这个环境中进化。D的目的仍是将真假区分开来，G需要经历不同的变异从而产生适应环境的后代。不同的训练损失函数最小化真假样本分布的距离，得到不同的变异。同时在给定当前D最优下衡量进化后的G生成样本的质量和多样性。最后据此保留进化后的G中表现好的继续训练、剔除表现差的。

### 进化算法

思想如上述

#### 变异 variation

种群中的每个个体（父代）G通过不同的变异算子生成一系列子代。

这些变异算子对应于不同的衡量真假分布距离的训练损失函数。有三个组成部分：
1）**minimax mutation** 最小最大算子。对应于原始GAN的损失函数。旨在最小化真假分布的JS散度。但存在梯度消失的问题。

![m](https://katherinaxxx.github.io/images/post/egan/minimax.jpg#width-full){:height="90%" width="90%"}

2）**heuristic mutation** 启发式算子。旨在最大化D被误导（生成样本被判为真）的对数概率。不像minimax mutation，在D拒绝生成样本时不会饱和（不等于0），因此可以避免梯度消失。但是，最小化heuristic mutation等价于最小化kl-2JS，在实际训练中可能会导致训练不稳定和生成质量波动。而且由于给生成样本较大惩罚导致模式崩溃

![h](https://katherinaxxx.github.io/images/post/egan/heuristic.jpg#width-full){:height="90%" width="90%"}

3）**least-squares mutation** 最小二乘算子。不饱和损失，且不会给极端惩罚，这样避免了梯度消失和一定成的上避免了模式崩溃。

![ls](https://katherinaxxx.github.io/images/post/egan/ls.jpg#width-full){:height="90%" width="90%"}


#### 评估 evaluation

每个子代通过适应度函数（依据环境D）进行衡量评估。

评估关注两点：一是质量，二是多样性。也就是说，G生成的生成样本要逼真且分布足够广泛以极大程度上避免模式崩溃。

![ls](https://katherinaxxx.github.io/images/post/egan/fq.jpg#width-full){:height="90%" width="90%"}

![ls](https://katherinaxxx.github.io/images/post/egan/fd.jpg#width-full){:height="90%" width="90%"}

![ls](https://katherinaxxx.github.io/images/post/egan/f.jpg#width-full){:height="90%" width="90%"}

F越大越好。关于多样性，当x或G(z)在D上得分较高时，D的梯度就要小，D变得光滑，这样就可以使生成样本分布更广泛从而避免模式崩溃。

#### 选择 selection

子代中适应度值差的会被删除，剩下会被保留进行下一轮进化迭代
选择方法是(μ,λ)-selection：当前子代总体（λ个）根据F得分排序，μ个最好的被选择进入下一轮。

### EGAN

按以上步骤每次进化完之后，D更新。整体算法如下所示

![al](https://katherinaxxx.github.io/images/post/egan/algorithm.jpg#width-full){:height="90%" width="90%"}

每次迭代，D先训练 $n_D$ 次，之后用进化算法生成子代（实际是每个父代个体在不同mutation下均要计算梯度更新参数得到子代）、计算适应度、选择，然后进行下一次迭代。

### 实验

> [code](https://github.com/WANG-Chaoyue/EvolutionaryGAN)

将EGAN与其他一些GANs对比，说明在大规模图像数据集上EGAN生成图像的表现更优。

#### 实现细节

用到的数据是模拟数据（混合高斯分布）和三个数据集（CIFAR-10, LSUN bedroom, and CelebA）。为了保证可比较性，模型结构保持一致，具体看论文。超参数的设置参考算法中给出的。还提到平衡质量和多样性的权重，经过grid search，给出参考模拟数据下取0.5，真实数据取0.001。

#### 评估指标

考虑采用用MMD、FID、IS作为量化指标。
其中，MMD用于衡量生成分布和真实分布的差异，但是很难用于高维图像数据；用到了IS（越大说明质量越好）、FID（越小说明两分布差异越小）

#### 模拟数据

模拟数据的实验表明多样性适应度得分可以衡量生成样本的多样性，并且抑制模式崩溃。

#### CIFAR-10

CIFAR-10上的实验表明在同样结构下，EGAN的评估指标均好于其他GANs，且训练step更少。虽然训练一次EGAN的复杂度更大，但是收敛需要的迭代次数更少，并给出了父代个数少于等于4这个参考。

#### LSUN

LSUN上的实验证明了EGAN结构的鲁棒性。与其他GANs对比可以发现，更改结构中的东西会对结果造成很大影响，而EGAN基本没有太大的影响。

### code

[源码](https://github.com/WANG-Chaoyue/EvolutionaryGAN)是基于theano的，准备着手做一个pytorch版本，日后附上。
