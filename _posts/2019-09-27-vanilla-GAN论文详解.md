---
layout: post
title: vanilla-GAN论文详解
date: 2019-09-27
Author: Katherinaxxx
tags: [GAN, generation model]
excerpt: "vanilla-GAN的介绍、理论及实现"
image: "/images/pic02.jpg"
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

>首先附上[vanilla-GAN论文](https://arxiv.org/abs/1406.2661 'vanilla-GAN论文')

## 生成模型

GAN（Generative Adversarial Networks）是无监督学习中生成模型的一种。所谓无监督学习就是指在训练过程中没有标签，而生成模型所做的就是用真实数据训练出一个生成器，使得这个生成器生成的数据与真实数据相似。相似实质上指的是真实数据和生成数据的分布相似。因而生成模型所做的实质上做的是参数估计的工作。

假设真实分布为$p_{data}$，生成器能生成分布为$p_{G}$的数据，目标就是使$p_{data}$和$p_{G}$越相似越好。假设$p_{data}$的参数为$\theta$，则只要求出$\theta$，那么$p_{data}(x,\theta)$就确定了。用极大似然估计（MLE）来估计参数$\theta$：

设$x_1,x_2,...,x_m$是来自$p_{data}$的样本。则$x_1,x_2,...,x_m$落在$p_{G}$的似然为
$$L=\Pi_{i=1}^{m}p_{G}(x_i,\theta)$$
二者分布越相似，$L$就要越大，则$\theta^*$

$$
\begin{aligned}
\theta^* &=arg\,\max_{\theta}L  \\
&=arg\,\max_{\theta}\Pi_{i=1}^{m}p_{G}(x_i,\theta)\\
&=arg\,\max_{\theta}\Pi_{i=1}^{m}logp_{G}(x_i,\theta) &(log为增函数不影响\theta取值)\\
& =arg\,\max_{\theta}\frac1m\sum_{i=1}^{m}logp_{G}(x_i,\theta) &(log性质；除以一个非负常数不改变\theta) \\
&\approx arg\,\max_{\theta}E_{x\backsim p_{data}}[logp_{G}(x,\theta)] &(样本均值是总体均值的估计) \\
& =arg\,\max_{\theta}[\int_x p_{data}(x)logp_{G}(x,\theta)\mathrm{d}x - \int_x p_{data}(x)logp_{data}(x)\mathrm{d}x] &(第二项与\theta无关) \\
& = arg\,\max_{\theta}[-KL(p_{data}||p_{G})] \\
& = arg\,\min_{\theta}KL(p_{data}||p_{G})
\end{aligned}
$$

$KL$散度衡量了两个分布的相似程度，越小说明分布越相似越。由上述推导可知，用MLE估计分布参数的过程可以转化成最小化$KL$散度的问题。

## vanilla-GAN
### 基本思想
GAN由生成器G和判别器D两部分组成，由于通常用于描述复杂数据的分布，所以D和G常用神经网络。G的目的是生成数据试图骗过D，D的目的是判断数据是来自于真实数据还是生成数据:

![v](https://katherinaxxx.github.io/images/post/GAN/v.jpg#width-full)

给定G，先最大化$V(G,D)$更新D，然后固定G，通过最小化$V(G,D)$更新G。算法如下：

![algorithm](https://katherinaxxx.github.io/images/post/GAN/algorithm.jpg#width-full)

G和D相互博弈最终会达到纳什均衡。大致训练过程如下：

![train](https://katherinaxxx.github.io/images/post/GAN/train.jpg#width-full)

图中，黑线表示真实数据的分布，绿线表示生成器生成数据的分布，蓝线是判别器的得分（经过sigmoid后得分越高说明判别器将判定为真实数据）。

训练最初（a)，G还不能生成“逼真”的数据、D还不稳定。经过一定训练（b），D比较稳定训练的比较好，可以准确地将真实数据和生成数据区分开来。（c）D反过来会指引G往真实数据分布的方向偏移。最终（d），G可以生成足够“逼真”的数据骗过D，而D也无法区分真实数据和生成数据。

因而，D和G其实不仅仅是“敌人”关系，也有“师生”关系，作为“老师”的D引导着G的学习。

### 理论证明
**1. 给定G，$D_G^*(x) = \frac{p_{data}(x)}{p_{data}(x)+p_{G}(x)}$**

*Proof.* 给定G，最大化$V(G,D)$，即

$$
\begin{aligned}
V(G,D) &= \int_xp_{data}(x)log(D(x))\mathrm{d}x+ \int_zp_{z}(z)log(1-D(G(z)))\mathrm{d}z \\
&= \int_xp_{data}(x)log(D(x))\mathrm{d}x + \int_xp_{G}(x)log(1-D(x))\mathrm{d}x  \\
&= \int_xp_{data}(x)log(D(x))+p_{G}(x)log(1-D(x))\mathrm{d}x  
\end{aligned}
$$

>第二个等号可以直观想，或者根据[Radon–Nikodym定理](https://en.wikipedia.org/wiki/Radon%E2%80%93Nikodym_theorem 'Radon–Nikodym定理')积分换元。

$$
\begin{aligned}
D^* &= arg\,\max_DV(G,D) \\
&= arg\,\max_D\int_xp_{data}(x)log(D(x))+p_{G}(x)log(1-D(x))\mathrm{d}x  \\
&= arg\,\max_Dp_{data}(x)log(D(x))+p_{G}(x)log(1-D(x)) \\
&\triangleq arg\,\max_Df(D)
\end{aligned}
$$
>第二个等号可以离散的看，每个点达到最大值则其积分上也最大，只要找到所有的最大值，函数就找到了。

$G$、$x$给定情况下¥$p_{data}(x)$、$p_G(x)$为常数。则对$f(D)$求导并令导数为0，即可得出

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x)+p_{G}(x)}$$
证毕
将$D^*(x)$带入$V(G,D)$,得到实质训练准则$C(G)$

$$
\begin{aligned}
C(G) &= max_DV(G,D) \\
&= E_{x\backsim p_{data}}[logD^*(x)] +E_{z\backsim p_G}[log(1-D^*(x))] \\
&= E_{x\backsim p_{data}}[log\frac{p_{data}(x)}{p_{data}(x)+p_{G}(x)}] +E_{z\backsim p_G}[log\frac{p_{G}(x)}{p_{data}(x)+p_{G}(x)}]
\end{aligned}
$$

正如上文提到的，生成模型最终目的是为了训练出好的生成器，GAN也是一样。因为我们更关心实质训练准则

**2.当且仅当$p_{data}=p_{G}$时，$C(G)$最小，且等于-log4**
*Proof.* 当$p_{data}=p_{G}$时，带入$C(G)$， 即可得$C(G)=-log4$
还需证明-log4为最小值，则对$C(G)$继续变形

$$
\begin{aligned}
C(G) &= E_{x\backsim p_{data}}[log\frac{p_{data}(x)}{p_{data}(x)+p_{G}(x)}] + E_{x\backsim p_G}[log\frac{p_{G}(x)}{p_{data}(x)+p_{G}(x)}]     \\
&= \int_xp_{data}(x)log\frac{p_{data}(x)}{p_{data}(x)+p_{G}(x)}\mathrm{d}(x)+\int_xp_G(x)log\frac{p_{G}(x)}{p_{data}(x)+p_{G}(x)}\mathrm{d}(x) \\
&= \int_xp_{data}(x)log\frac{p_{data}(x)}{\frac{p_{data}(x)+p_{G}(x)}{2}}\mathrm{d}(x)+\int_xp_G(x)log\frac{p_{G}(x)}{\frac{p_{data}(x)+p_{G}(x)}{2}}\mathrm{d}(x)-2log2 \\
&= KL(p_{data}||\frac{p_{data}+p_{G}}{2}) + KL(p_{G}||\frac{p_{data}+p_{G}}{2}) - log4 \\
&= 2JS(p_{data}||p_{G}) -log4
\end{aligned}
$$

由于$JS$散度大于零，故$C(G)$最小值为-log4，证毕。

由以上证明，可以看出最终GAN也是要找一个最优的$G^*$使得$C(G)$最小，即也是使$JS$散度最小。与开篇联系起来了。


### 总结
GAN的绝妙的对抗思想引发了学术界的思潮，但其本身在训练上有许多需要攻克的难题。因此在vanilla GAN的基础上衍生出了很多GANs用于解决这些问题，在此不赘述。

### python实现
最后附上GAN基于tensorflow的[实现](https://github.com/Katherinaxxx/MyML/blob/master/lib/generator/GAN.py '实现')。
