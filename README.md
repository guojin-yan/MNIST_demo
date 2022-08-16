# 手写数字识别

## 1. 项目简介

手写数字识别是指给定一系列的手写数字图片以及对应的数字标签，构建模型进行学习，目标是对于一张新的手写数字图片能够自动识别出对应的数字。图像识别是指利用计算机对图像进行处理，通过模型对其分析和理解，得到图片文件中所写的数字。

在人工智能领域，手写数字识别被问题转换为自动分类问题。将0~9之内的10个数字分为10类，通过模型训练，实现对数字图片的分类，间接获取数字图片上的手写数字。

## 2. 数据集介绍

MNIST数据集是一个公开手手写数字识别数据集，该数据集由250个不同的人手写而成，总共有7000张手写数据集。其中训练集有6000张，测试集有1000张。每张图片大小为28x28，为处理后的灰度图，是由28x28个像素点组成。

![Figure_1](E:\Git_space\手写数字识别\image\Figure_1.png)

上图为手写数字数据集中的部分图片。该数据集可以通过以下路径进行下载：[MNIST](http://yann.lecun.com/exdb/mnist/) (http://yann.lecun.com/exdb/mnist/) ；或者通过各种深度学习框架提供的API函数进行下载。

![image-20220814175037775](E:\Git_space\手写数字识别\image\image-20220814175037775.png)

通过官网下载的方式需要分别下载下图中的四个链接对应的文件，下载完成后，将文件解压到本地即可。

![image-20220814175204076](E:\Git_space\手写数字识别\image\image-20220814175204076.png)

下图为解压好的文件，该文件为处理后的二进制文件，不是现成的图片文件，不可以直接打开，需要进行处理才可以读取，后面会在Matlab训练手写数字识别模型处详细讲解该文件的读取方式。

![image-20220814175308978](E:\Git_space\手写数字识别\image\image-20220814175308978.png)

## 3. 数据集文件读取

数据集文件主要分两种：一种是图片数据文件，一种是分类标注文件。文件为二进制文件格式。

以训练集文件为例：``train-images-idx3-ubyte``，该文件为保存的二值化后的手写数字图片数据，大小为28×28×1。我们通过Matlab读取数据文件：

```matlab
% 打开二进制文件
fid = fopen('train-images-idx3-ubyte', 'rb');
% 读取二进制文件，数据格式为uint8，将所有数据读取到train_images_data中
train_images_data = fread(fid, inf, 'uint8', 'l');
% 前16个数据去掉
train_images_data = train_images_data(17:end);
% 关闭文件
fclose(fid);
% 将数据矩阵转为28×28×60000
train_images = reshape(train_images_data,28,28,60000);
% 交换前两维度，不然图片是反的
train_images = permute(train_images,[2 1 3]);
% 将数据维度转为28×28×1×60000
train_X(:,:,1,:) = train_images;
```

经过上面读取后，我们可以获得28×28×1×60000的数据矩阵，其中28×28×1为一张图片，总共有60000张图片。

下面我们读取训练集的标注文件：``train-labels-idx1-ubyte`` 存储了训练集的标注情况。

```matlab
% 文件读取方式与上面一致
fid = fopen('dataset\train-labels-idx1-ubyte', 'rb');
train_labels = fread(fid, inf, 'uint8', 'l');
% 标注文件是从第9个开始读取，与上面图片数据的顺序一致
train_labels = train_labels(9:end);
fclose(fid);
```

如果我们想通过Python实现数据文件读取

```
import numpy as np
import struct
 
from PIL import Image
import os
```





```python
# 根目录
home_path = 'E:\Git_space\手写数字识别\Datasets'

# 图片文件地址
image_file =  os.path.join(home_path, 'MNIST\\raw\\train-images-idx3-ubyte') 

# 数据文件大小为：28×28×1×60000+16 = 47040016
image_data_size = 47040016
# 有效文件数据为： 28×28×1×60000 = 47040000
image_data_size = str(image_data_size - 16) + 'B'
# 打开文件
image_data_buffer = open(image_file, 'rb').read()
# 获取图片缓冲内存数据中的图片数量、行数、列数
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', image_data_buffer, 0)
# 读取图片文件的数据
image_datas = struct.unpack_from('>' + image_data_size, image_data_buffer, struct.calcsize('>IIII'))
# 将图片数据转为uint8格式，转为[numImages, 1, numRows, numColumns]大小的矩阵
image_datas = np.array(image_datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)
 
# 标签文件地址
label_file = os.path.join(home_path, 'MNIST\\raw\\train-labels-idx1-ubyte' )

# 标签文件长：60000+8 = 60008
label_data_size = 60008
# 实际标签文件长：60000+8 = 60008
label_data_size = str(label_data_size - 8) + 'B'
 
label_data_buffer = open(label_file, 'rb').read()
magic, numLabels = struct.unpack_from('>II', label_data_buffer, 0)
label_datas = struct.unpack_from('>' + label_data_size, label_data_buffer, struct.calcsize('>II'))
label_datas = np.array(label_datas).astype(np.int64)
```





## 4. Python基于Pytorch框架实现模型训练

### 4.1 训练环境

CUDA 11.4

Pytorch 1.12.1+cu113

Python 3.9



### 4.2 定义数据加载器

新建一个数据加载器文件``dataloader.py``，用于加载训练数据，后续模型训练时通过数据加载器按批次加载训练集。

首先导入以下模块，该模块在安装``torch``模块后就可以使用。

```python
import torchvision
from torch.utils.data import DataLoader
```

接下来定义一个``class Dataloader()``类，并初始化相关变量。主要设置训练集、测试集批次大小即可，后面我们直接调用Pytorch的MNIST数据集 API 接口加载数据集，所以需要定义的变量较少。

```python
'''
    数据集加载类
    功能：
    实现数据集本地读取，并将数据集按照指定要求进行预处理；
    后续模型训练直接调用DataLoader逐步进行训练
    初始化参数：
    batch_size_train：训练集批次
    batch_size_test：测试集批次
'''
class Dataloader():
    def __init__(self,batch_size_train, batch_size_test):
        # 初始化训练集bath size
        self.batch_size_train = batch_size_train
        # 初始化测试集bath size
        self.batch_size_test = batch_size_test
   
```

分别定义训练集加载器``train_loader()``以及``test_loader()``，用于加载训练集以及测试集。``torchvision.datasets.MNIST()``是Pytorch提供的MNIST数据集加载接口API函数：``'./Datasets/'``为指定的数据集本地下载路径；``download``可以设置是否下载数据集，当指定为``True``且首次运行时，会将数据集下载到指定的路径下，再次运行时会检测路径下是否有该文件，如果有会直接读取，如果没有将会重新下载；``transform``指定的为数据处理方式，主要是将数据进行归一化以及数据类型转换处理，还可以对数据进行增强处理；``batch_size``指定训练时数据集加载的批次大小。

```python
#加载训练集
def train_loader(self):
    # 调用pytorch自带的DataLoader方法加载MNIST训练集；
    # 直接使用pytorch的MNIST数据集API接口加载数据，
    # 第一次使用可以设置为True
    train_load = DataLoader(
        torchvision.datasets.MNIST('./Datasets/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
        batch_size = self.batch_size_train, shuffle=True)
    return train_load
```

```python
#加载测试集
def test_loader(self):
    # 调用pytorch自带的DataLoader方法加载MNIST测试集
    test_load = DataLoader(
        torchvision.datasets.MNIST('./Datasets/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
        batch_size = self.batch_size_test, shuffle=True)
    return test_load
```

### 4.3 定义网络

接下来定义训练网络，要实现后面我们的网络能够很好的识别我们的手写数字，在此处就要定义一个比较好的网络，下面的网络是我参考的网上一些人所做的模型定义的。

首先导入一下模块：``torch.nn ``模块下定义了各种我们常见的网络层以及网络结构，直接调用该模块进行网络构建；``torch.nn.functional``模块是定义了各种激活函数的模块。

```python
import torch.nn as nn
import torch.nn.functional as F
```

