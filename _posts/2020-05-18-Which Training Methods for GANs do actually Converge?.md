---
layout: post
title: Which Training Methods for GANs do actually Converge?
date: 2020-05-18
Author: Katherinaxxx
tags: [GAN, generation model]
excerpt: "vanilla-GAN的介绍、理论及实现"
image: "/images/post/.jpg"
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

>首先附上[Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406)
[参考](https://blog.csdn.net/w55100/article/details/88091704)

* any list
{:toc}

>由于最近要开始做课题GAN相关理论部分，查到了这篇论文，发于ICML2018，学习膜拜一下
## 一步步开始读论文吧

### Abstract
已有研究表明，绝对连续的数据和生成器可以使得GAN在训练时局部收敛。本文则说明绝对连续是必要条件：举反例，不绝对连续更实际的数据下，无惩罚的GAN不一定局部收敛。

此外，我们的分析表明，具有实例噪声或零中心梯度惩罚的GAN训练收敛。 另一方面，我们证明每个生成器更新具有有限数量的鉴别器更新的Wasserstein-GAN和WGAN-GP并不总是收敛于平衡点。

根据我们的分析，我们将收敛结果扩展到更通用的GAN，并证明了局部收敛，以简化梯度惩罚，即使生成器和数据分布位于较低维流形上。

### 1. Introduction

首先，总结了一些关于研究GAN训练的文献。

Recently, Mescheder et al. (2017) and Nagarajan & Kolter (2017)---可以通过检查联立梯度向量场的雅可比矩阵的特征值来分析GAN训练的局部收敛性和稳定性：如果Jacobian矩阵在平衡点上仅具有负实部的特征值，则学习率足够小时，GAN训练会在局部收敛。 另一方面，如果雅可比矩阵的特征值有虚轴部分，则通常不会局部收敛。

*PS.相关数学概念
梯度向量场：[百度百科](https://baike.baidu.com/item/%E6%A2%AF%E5%BA%A6%E5%90%91%E9%87%8F%E5%9C%BA),这里的联立梯度向量场就是指分别对GD参数求梯度得到的向量联合起来。
雅各比矩阵：函数一阶偏导数组成的矩阵。
简单理解一下，GAN的训练中最小化损失，因而负实部特征值似乎可以保证向下的趋势？*

Mescheder(2017）表明，如果特征值接近但不在虚轴上，则训练算法可能需要极小的学习率才能实现收敛。而Mescheder(2017在实践中观察到的接近于虚轴的特征值，该观察结果并未回答以下问题：接近虚轴的特征值是否是普遍现象；如果是，则它们是否确实是训练不稳定的原因。

Nagarajan＆Kolter（2017）给出了对该问题的部分答案，他表明，对于绝对连续的数据和生成器分布，雅可比行列式的所有特征值均具有负实部。因此，在这种情况下，GAN的收敛速度足够小即可收敛。但是，（Sønderby et al., 2016; Arjovsky & Bottou, 2017）指出对于GAN的常见用例而言，绝对连续性的假设并不正确，因为这两种分布都可能位于较低维的流形上。

*PS.直观理解，高维空间的低维流形极有可能是不连续的*

ooooooh :)

然后，论文表明绝对连续这一假设是必要的，通过典型的例子表明未正则化的GAN不总能局部收敛；同时WGAN、WGAN-GP、DRAGAN[正则化的GAN]在G更新一次D更新固定次数下不收敛；另一方面，实例噪声（Sønderby2016; Arjovsky＆Bottou，2017），零中心梯度惩罚（Roth2017）和一致优化（Mescheder2017）导致局部收敛。同样，提出了自己的正则化/惩罚方法，能生成分辨率高的图像且不怎么需要调参

*PS.实例噪声：在真实数据和生成数据中都添加噪声，使得二者有不可忽略的重叠部分
零均值GP：每个batch的梯度减去梯度的均值，可以限制1-lipschitz*

### 2. Instabilities in GAN training
#### 2.1 Background
介绍了一下描述了一下均衡点的符号表示以及GAN训练一个通用的形式
#### 2.2 The Dirac-GAN

通过一个例子表明，一般情况下无惩罚的GAN既不会局部收敛也不会全局收敛。

首先，定义Dirac-GAN，在这种情况下，DG均只有一个参数，且真实分布是以0为中心的的狄拉克分布，因此（1）可以写成（4）

【此处添加方程】

？？？figure1没看懂 （4）怎么写成JS散度 为啥同样适用于wasserstein距离这一点说是3.1节会讲

Lemma2.2意思是均衡点的梯度向量场的雅各比的两个特征值均在虚轴上。
这意味着通常不会以线性收敛于均衡点，但仍有可能以亚线性收敛。？？？why

Lemma2.3说梯度向量场的积分曲线不收敛到那是均衡点。
后面的结论没看懂

接着看离散情况

Lemma2.4证明了同时梯度下降下，更新算子的雅各比矩阵在纳什均衡的特征值公式。这意味着同时梯度下降在均衡点附近不是稳定的；更有甚者，迭代参数的范数是单调递增的？？不知道怎么推出来的

Lemma2.5证明了交替梯度下降下，更新算子的雅各比矩阵的特征值公式
意味着虽然交替更新的情况下不能以线性速度收敛到纳什均衡，但原则上可以亚线性速度收敛。但是正如Lemma2.3指出，即使连续都不收敛，而且实验发现总是稳定围绕在均衡点附近并且没有收敛的趋势。

【此处fig2b】

【我如果要看三个是否能收敛 可以参考他这里的证明】

#### 2.3 Where do instabilities come from?

GAN的训练过程采用的是交替训练，希望D和G最终达到纳什均衡。然而这其实是一个非常理想的状态，因为在实际训练过程中很难控制二者强弱，比如，训练最初G远离真实分布的时候，D会促使G向真实分布靠近，然后随着D训练更好，D又会将G推离真实分布，因此会处于一个震荡的状态。并且【Which Training Methods for GANs do actually Converge?】实验表明原始GAN这种训练方法会最终以圆型围绕在纳什均衡周围，并且没有进一步收敛的趋势。

针对训练不稳定的问题，Nagarajan & Kolter (2017)表明在稳定假设下，即绝对连续分布，用梯度优化的GAN可以局部收敛。然而，这一假设又是很难满足的，因为GAN的应用场景基本是在图像方面。然而，即使数据分布是绝对连续的，但集中在某些低维流形上，梯度矢量场的雅可比矩阵的特征值也将非常接近虚轴，从而导致病态严重的问题Mescheder et al. (2017)。

### 3. Regularization strategies

这部分考虑不同惩罚策略对收敛的影响或者作用。
归纳了一幅图，非常清晰。
【此处fig3】
#### 3.1 Wasserstein GAN

如果数据分布和生成器分布的支撑集不匹配，则JS散度可能相对于生成器的参数是不连续的，甚至可以取无穷大的值。

为了使散度对于的生成器参数连续，Wasserstein GAN（WGAN用Wasserstein-divergence代替了GANs的原始推导中使用的Jensen-Shannon散度。并且施加Lipschitz连续的惩罚。如果鉴别器始终经过训练直到收敛，则WGAN会收敛，但实际上WGAN通常通过每次生成器更新仅运行固定数量的鉴别器更新来进行训练。

Lemma3.1证明了上述说的，WGAN在同时或者交替梯度优化策略下，更新一次G更新固定次数D，通常不会收敛到纳什均衡。

#### 3.2 Instance noise

尽管最初的动机是使数据和生成器分布之间的概率差异得到明确定义，即没有共同的支撑集，但这并未阐明实例噪声对训练算法本身的影响及其找到纳什均衡的能力。

Lemma3.2证明了用实例噪声时，梯度向量场的雅各比矩阵的特征值公式。从而可以看出在同时或者交替下均局部收敛。

从fig3可以看出，实例噪声确实在梯度矢量场中创建了一个很强的径向分量，这使得训练算法收敛。


#### 3.3. Zero-centered gradient penalties

Lemma3.3证明了用零均值梯度惩罚时，梯度向量场的雅各比矩阵的特征值公式。从而可以看出在同时或者交替下均局部收敛。

### 4. General convergence results

这部分讲的就是本文提出的两种惩罚。适用于通常的GAN，并且可以证明即使不满足Assumption IV in Nagarajan & Kolter (2017)，也可以推广到他的收敛证明

#### 4.1. Simplified gradient penalties

分析表明，零中心梯度惩罚关于局部稳定的主要影响是，惩罚判别器偏离纳什均衡。 实现此目的的最简单方法是仅对真实数据进行梯度惩罚：当生成器分布产生真实数据分布，并且数据流形上的判别器等于0时，梯度惩罚可确保判别器无法产生与数据流形正交而不会在GAN博弈中遭受损失的非零梯度。

？？？意思是不是说产生0梯度 从而在G分布与真实分布一致时 不会继续更新 也就不会再把G推离真实分布。

随后给出了这种惩罚的公式，两种
【此处公式（9）（10）】

#### 4.2. Convergence

这一部分证明在一定稳定假设下，提出的惩罚方法的收敛结果。

















































## 读后感 :)
