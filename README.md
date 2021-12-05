# Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data


## 目录

```
1. 简介
2. 数据集和复现精度
3. 开始使用
4. 代码结构与详细说明
```

## 1. 简介
![Real-ESRGAN](https://user-images.githubusercontent.com/52402835/144571624-a29d9a88-d08a-4891-8356-2d9d62798774.jpg)

本项目基于深度学习框架PaddlePaddle对Real-ESRGAN网络进行复现。Real-ESRGAN网络属于生成对抗网络，包括基于ESRGAN的生成器和基于U-Net的判别器，可对真实世界的复杂图像进行超分辨率重建。


**论文:** [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://paperswithcode.com/paper/real-esrgan-training-real-world-blind-super)

**参考repo:** [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

在此非常感谢` Xintao `贡献的[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)的`Pytorch`代码，提高了本repo复现论文的效率。

**aistudio体验教程:** [地址](https://aistudio.baidu.com/aistudio/projectdetail/3156903)


## 2. 数据集准备

本项目训练所用的数据集为[DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)，[Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)和[OST](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip)。

|数据集|大小|下载链接|数据格式|
| :---: | :---: | :----: |:----: |
|DIV2K|120k|[DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)|`png`|
|Flickr2K|120k|[Flickr](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)|`png`|
|OST|120k|[OST](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip)|`png`|

## 3. 复现效果
基于上述数据集的训练结果，在3张图片上的测试结果如下：


![test_image1](inputs/00003.png)![test_image2](inputs/0014.jpg)![test_image3](inputs/0030.jpg)
基于上述数据集，本repo复现的精度、数据集名称、模型下载链接（模型权重和对应的日志文件推荐放在**百度云网盘**中，方便下载）、模型大小，以表格的形式给出。如果超参数有差别，可以在表格中新增一列备注一下。

如果涉及到`轻量化骨干网络验证`，需要新增一列骨干网络的信息。



## 3. 开始使用

### 3.1 准备环境

- 硬件：Tesla V100 GPU
- 框架：PaddlePaddle >= 2.2.0


**克隆本项目**
```
# clone this repo
git clone https://github.com/20151001860/Real_ESRGAN_paddle.git
cd Real_ESRGAN_paddle
```
**安装第三方库**
```
pip install -r requirements.txt
```
然后介绍下怎样安装PaddlePaddle以及对应的requirements。

建议将代码中用到的非python原生的库，都写在requirements.txt中，在安装完PaddlePaddle之后，直接使用`pip install -r requirements.txt`安装依赖即可。


### 3.2 快速开始

**训练**


为了训练`Real-ESRGAN`模型，我们采用与原论文一致的初始化模型参数`ESRGAN_SRx4_DF2KOST_official-ff704c30.pth`，并将其转化为Paddle格式的权重`ESRGAN_SRx4_DF2KOST_official-ff704c30.pdparams`进行训练。
```
python train.py --opt options/train_realesrgan_x4plus.yml
```
训练保存的模型和日志可见：

**测试**

```
python inference_realesrgan.py
```
测试使用7张图片，结果如下：


## 4. 代码结构

```
├─data                          
├─datasets                         
├─experiments                           
├─inputs       
├─loss
├─models
├─options
├─results
├─tb_logger
├─utils
│  generate_meta_info.py                     
│  inference_realesrgan.py                        
│  README.md                        
│  README_CN.md                     
│  requirements.txt                                       
│  train.py                                     

```
需要用一小节描述整个项目的代码结构，用一小节描述项目的参数说明，之后各个小节详细的描述每个功能的使用说明。

## 5.模型信息
相关信息:

| 信息 | 描述 |
| --- | --- |
| 作者 | 勇敢土豆不怕困难！|
| 日期 | 2021年12月 |
| 框架版本 | PaddlePaddle==2.2.0 |
| 应用场景 | 超分辨率重建 |
| 硬件支持 | GPU、CPU |
| 在线体验 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/3156903)|
