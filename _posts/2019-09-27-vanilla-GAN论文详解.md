---
layout: post
title: vanilla-GAN论文详解
date: 2019-09-27
Author: Katherinaxxx 
tags: [GAN, generation model]
comments: true
toc: true
---

首先附上[vanilla-GAN论文](https://arxiv.org/abs/1406.2661 'vanilla-GAN论文')

# 生成模型

GAN（Generative Adversarial Networks）是无监督学习中生成模型的一种。所谓无监督学习就是指在训练过程中没有标签，而生成模型所做的就是用真实数据训练出一个生成器，使得这个生成器生成的数据与真实数据相似。相似实质上指的是真实数据和生成数据的分布相似。因而生成模型所做的实质上做的是参数估计的工作。

