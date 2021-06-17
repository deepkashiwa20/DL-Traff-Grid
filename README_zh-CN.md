# DL-Traff-Graph: Graph-Based Deep Learning Models for Urban Traffic Prediction

## Introduction
[English](README.md) | 简体中文

DL-Traff是一个开放资源项目，为基于网格和基于图形的模型的流量预测提供了基准。DL-Traff-grid是基于格子的项目的一部分。这部分主要工作在TensorFlow/tensorflow-gpu (= 1.1x.0) 和 Keras (>= 2.0.8)上。在这个github中，我们将一些基于格子的网络模型集成到一个平台中。我们保证了所有模型都基于相同的数据处理、相同的超参数和相同的计算环境，如Pytorch和Cudnn的版本。尽管这会使得各个模型没有达到最终收敛的效果，但是正因如此而可以充分体现不同网络架构在同条件下的表现性能。我们会在后续工作中更新各个模型调优后的结果。
## 安装依赖环境
工作环境和主要依赖包:
* Ubuntu 16.04.6 LTS
* Python 3 (>= 3.5; Anaconda Distribution)
* NumPy (>= 1.11.0)
* pandas (>= 0.18.0)
* TensorFlow/tensorflow-gpu (= 1.1x.0)
* Keras (>= 2.0.8)

## 公开数据集和模型库
### 数据集
* TaxiBJ
* BikeNYC-I
* BikeNYC-II
* TaxiNYC

### 模型
* CNN
* ConvLSTM
* STResNet
* PCRN
* DMVST-Net
* DeepSTN+
* STDN

## 组成介绍和用户指导

### 目录
* BikeNYC1  (数据集文件夹)
  * day_information_onehot.csv  (one-hot特征文件)
  * flowioK_BikeNYC1_20140401_20140930_60min.npy  (特征数据文件, 按照数据集名字，总时间范围，统计周期来命名)
  * ...
* BikeNYC2
  * ...
* TaxiBJ
  * ...
* TaxiNYC
  * ...
* workBikeNYC1/predflowio  (BikeNYC1 数据集下的主程序)
  * parameter.py  (共同参数文件, 提供每个模型都会用到的参数)
  * Param_DeepSTN_flow.py (模型独有参数文件,提供仅限于本模型的参数。如果出现了和parameter.py一样的参数，本文件的参数将有优先权。)
  * load_data.py  (数据读取模块)
  * preprocess_flow.py  (模型训练之前的交通流预处理)
  * DeepSTN_net.py  (模型文件, 用来debug以及提供模型网络给主程序调用)
  * predflowio_DSTN+.py (主程序, 针对DeepSTN_net网络的训练预测测试文件)
  * ...
* workBikeNYC2/predflowio (main program folder in BikeNYC2 dataset)
  * ...
* workTaxiBJ/predflowio (main program folder in TaxiBJ dataset)
  * ...
* workTaxiNYC/predflowio (main program folder in TaxiNYC dataset)
  * ...
  * 
### 用户指导
下载源码到你的设备上, 当你进入到一个路径后输入一下指令，代码将会被下载到该路径下:
```
git clone git@github.com:deepkashiwa20/DL-Traff-Grid.git
```

* debug和运行模型
* 
用DeepSTN_net在BikeNYC1数据集下的运行来示范使用方法：

```
cd /workMETR

# 在1号显卡上debug模型 :
python DeepSTN_net.py 1

# 在1号显卡是运行主程序进行训练，预测和测试。
python predflowio_DSTN+.py 1

```
