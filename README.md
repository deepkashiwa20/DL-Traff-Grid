# [CIKM 2021 Resource Paper] DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction
## DeepTraffic-Grid: Grid-Based Deep Learning Models for Urban Traffic Prediction

* Our work has been accepted by CIKM 2021 Resource Track. https://doi.org/10.1145/3459637.3482000
* The preprint version has been uploaded to arXiv. https://arxiv.org/pdf/2108.09091.pdf
* The url of Graph-Based work is : (https://github.com/deepkashiwa20/DL-Traff-Graph)

## Cite
@inproceedings{10.1145/3459637.3482000,
author = {Jiang, Renhe and Yin, Du and Wang, Zhaonan and Wang, Yizhuo and Deng, Jiewen and Liu, Hangchen and Cai, Zekun and Deng, Jinliang and Song, Xuan and Shibasaki, Ryosuke},
title = {DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction},
year = {2021},
isbn = {9781450384469},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = { https://doi.org/10.1145/3459637.3482000 },
doi = {10.1145/3459637.3482000},
booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
pages = {4515–4525},
numpages = {11},
location = {Virtual Event, Queensland, Australia},
series = {CIKM '21}
}

## Introduction
English | [简体中文](README_zh-CN.md)

DL-Traff is an open resourse project which offers a benchmark for traffic prediction on grid-based and graph-based models. DL-Traff-Grid is a part of grid-based project. This main branch works on TensorFlow/tensorflow-gpu (= 1.1x.0) and Keras (>= 2.0.8). In this github, we integrate several grid-based models into one platform. We maintain that all models are based on the same data processing, the same hyperparameters, and the same computing environment such as the version of tensorflow and Cudnn. Although this makes the models fail to achieve the final convergence effection, the performance of different network architectures under the same conditions will be fully reflected by our experiment. We will update the optimization results of each model in later work.

## Installation Dependencies
Working environment and major dependencies:
* Ubuntu 16.04.6 LTS
* Python 3 (>= 3.5; Anaconda Distribution)
* NumPy (>= 1.11.0)
* pandas (>= 0.18.0)
* TensorFlow/tensorflow-gpu (= 1.1x.0)
* Keras (>= 2.0.8)

## Public data and models zoo
### Datasets
* TaxiBJ
* BikeNYC-I
* BikeNYC-II
* TaxiNYC

### Models
* CNN
* ConvLSTM
* STResNet
* PCRN
* DMVST-Net
* DeepSTN+
* STDN

## Components and user guide

### Content
* BikeNYC1  (dataset folder)
  * day_information_onehot.csv  (one-hot feature file)
  * flowioK_BikeNYC1_20140401_20140930_60min.npy  (feature data, named by dataset name, time range and sample interval)
  * ...
* BikeNYC2
  * ...
* TaxiBJ
  * Please first use unzip cmd to unzip TaxiBJ13.zip~TaxiBJ16.zip to the same folder.
* TaxiNYC
  * ...
* workBikeNYC1/predflowio  (main program folder in BikeNYC1 dataset)
  * parameter.py  (common parameter file, which provide the parameters every model will use)
  * Param_DeepSTN_flow.py  (Model-specific parameter file, which provide the parameters this model will use. If the same parameters as parameter.py appear, this file has priority.)
  * load_data.py  (load data module)
  * preprocess_flow.py  (flow preprocessing before model training)
  * DeepSTN_net.py  (model file, used for debug and providing model interfaces for pred programs.)
  * predflowio_DSTN+.py (pred file, used for train, prediction and test of the single model DeepSTN)
  * ...
* workBikeNYC2/predflowio (main program folder in BikeNYC2 dataset)
  * ...
* workTaxiBJ/predflowio (main program folder in TaxiBJ dataset)
  * ...
* workTaxiNYC/predflowio (main program folder in TaxiNYC dataset)
  * ...
### User guide
Download this project into your device, the code project will be downloaded into the current path where you type this powershell command:
```
git clone git@github.com:deepkashiwa20/DL-Traff-Grid.git
```

Use the DeepSTN model on BikeNYC1 dataset as an example to demonstrate how to use it. 

* debug and run model
```
cd /workBikeNYC1

# Debug the model on video card 1 :
python DeepSTN_net.py 1

# Run the main program to train, prediction and test on video card 1:
python predflowio_DSTN.py 1

```



