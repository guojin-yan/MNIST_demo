# 手写数字识别

## 1. 项目简介

手写数字识别是指给定一系列的手写数字图片以及对应的数字标签，构建模型进行学习，目标是对于一张新的手写数字图片能够自动识别出对应的数字。图像识别是指利用计算机对图像进行处理，通过模型对其分析和理解，得到图片文件中所写的数字。

在人工智能领域，手写数字识别被问题转换为自动分类问题。将0~9之内的10个数字分为10类，通过模型训练，实现对数字图片的分类，间接获取数字图片上的手写数字。

该项目所用到的源码以及所有源码均在GitHub以及Gitee上面开源，下载方式：

```shell
GitHub: 
git clone https://github.com/guojin-yan/MNIST_demo.git

Gitee:
git clone https://gitee.com/guojin-yan/MNIST_demo.git
```



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

如果我们想通过Python实现数据文件读取，首先导入以下模块：

```
import numpy as np
import struct
 
from PIL import Image
import os
```

下面为训练集图片数据以及标签数据读取方式：

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

测试集读取方式类似，如有需要自行修改。



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

### 4.3 定义网络（net,py）

接下来定义训练网络，要实现后面我们的网络能够很好的识别我们的手写数字，在此处就要定义一个比较好的网络，下面的网络是我参考的网上一些人所做的模型定义的。

首先导入一下模块：``torch.nn ``模块下定义了各种我们常见的网络层以及网络结构，直接调用该模块进行网络构建；``torch.nn.functional``模块是定义了各种激活函数的模块。

```python
import torch.nn as nn
import torch.nn.functional as F
```

我们将网络定义到Net类中，继承``nn.Module`''类模板。在初始化时定义一些在前向传播中所用到的网络层，方便再后面直接使用。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''定义相关的计算层'''
        # 定义一个卷积核为1×10的卷积层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 定义一个卷积核为10×20的卷积层
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 定义一个二维的Dropout2层，防止模型训练时过拟合
        self.conv2_drop = nn.Dropout2d()
        # 定义全连接层
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
```



我们在此处模拟网络的前向传播，定义一个简单的网络，有两步卷积运算以及两步全连接组成。卷积运算有卷积->池化->激活三步组成，其中为了防止网络出现过拟合，增加了Dropout层。

```python
    def forward(self, x):
        # 一次卷积运算
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # 二次卷积运算
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        # 设置数据长度
        x = x.view(-1, 320)
        # 一次全连接
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # 二次全连接
        x = self.fc2(x)
        # 分类模型调用分类结果处理函数
        return F.log_softmax(x) 
```

以下模型结构图为训练完成的模型转为ONNX模型的结构图。

**![image-20220816213329067](E:\Git_space\手写数字识别\image\image-20220816213329067.png)**



### 4.4 定义训练器(trainer.py)

训练器主要是将构建好的模型以及训练集进行训练，实现对训练过程数据的记录以及训练模型的保存。在此处我们将其封装到``trainer.py``文件中``Trainer()``类中。

首先导入以下模块:

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import time
```

``tensorboard``是一个外部插件，用于保存训练过程的数据并进行可视化显示，在使用时需要自行安装，他不在Pytorch软件包中。

接下来就是创建``Trainer()``类，在该类的初始化方法中，我们主要实现了对一些成员变量进行赋值，并且初始化后面训练的相关成员变量。

```python
class Trainer():
    def __init__(self, network, learning_rate, log_interval):
        # 初始化网络
        self.network = network
        # 初始化学习率
        self.learning_rate = learning_rate
        # 初始化优化器
        self.optimizer = optim.SGD(network.parameters(), lr=learning_rate)
        # 初始化日志打印间隔
        self.log_interval = log_interval
        # 初始化损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        # 当CUDA可用时，开启CUDA加速
        if torch.cuda.is_available():
              self.loss_fn =   self.loss_fn.cuda()
        # 初始化tensorboard日志保存接口
        self.writer = SummaryWriter("Python/logs_train")
        # 初始准确率
        self.accuracy = 0
```

下面为模型训练方法，主要实现模型训练、模型测试、日志保存与打印、模型保存与打印等功能。具体过程可以根据代码解释进行理解。

```python
    def tarin(self, epoch, train_loader, test_loader):
        # 训练和测试的总步数
        total_train_step = 0
        total_test_step = 0
        # 获取当前时间
        now =  time.localtime()
        # 创建日志
        log_text=open("Python/logs_train/{}.txt".format(time.strftime("%Y_%m_%d_%H_%M", now)),mode='w')
        log_text.write("模型训练时间：{}".format(time.strftime("%Y-%m-%d %H:%M:%S", now)))
        log_text.write('\r\n')
        for i in range(epoch):
            # 模型训练
            print("--------第{}轮训练开始--------".format(i))
            log_text.writelines("--------第{}轮训练开始--------".format(i))
            log_text.write('\r\n')
            self.network.train()
            '''
                模型训练步骤：
                1.在DataLoader中读取bath_size个数据，包括模型输入和目标输出
                2.带入到网络中计算
                3.带入损失函数，计算损失
                4.模型反向传播
                5.执行单个优化步骤
                6.重复上面过程，反复训练
            '''
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                # 当可以使用CUDA加速时，将data、target转为CUDA格式
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                # 带入网络计算，前向传播
                output = self.network(data)
                # 带入损失函数计算
                loss = self.loss_fn(output,target)
                # 反向传播
                loss.backward()
                # 执行单个优化步骤
                self.optimizer.step()
                total_train_step +=1
                if batch_idx % self.log_interval == 0:
                    # 打印训练过程日志
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
                    # 写入txt日志文件
                    log_text.writelines('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
                    log_text.write('\r\n')
                    # 将训练结果写入tensorboard日志中
                    self.writer.add_scalar("train_loss",loss.item(),total_train_step)
            #模型测试
            print("--------第{}轮测试开始--------".format(i))
            log_text.write("--------第{}轮测试开始--------".format(i))
            log_text.write('\r\n')
            total_test_loss = 0
            total_accuracy = 0
            accuracy = 0
            with torch.no_grad():# with以下的代码不会更改
                for batch_idx, (data, target) in enumerate(test_loader):
                    # 当可以使用CUDA加速时，将data、target转为CUDA格式
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()
                    # 带入模型计算
                    outputs = self.network(data)
                    # 带入损失函数计算下损失
                    loss = self.loss_fn(outputs,target)
                    # 计算总损失
                    total_test_loss = total_test_loss + loss.item()
                    # 计算准确率
                    accuracy = (outputs.argmax(1) == target).sum()
                    # 总的准确率
                    total_accuracy = total_accuracy + accuracy
            accuracy = total_accuracy / len(test_loader.dataset)
            # 保存结果最好的模型
            if(self.accuracy < accuracy):
                torch.save(self.network, "Python/best_model.pth".format(i))
                print("best_model.pth 保存成功")
                log_text.writelines("best_model.pth 保存成功")
                log_text.write('\r\n')
            self.accuracy = accuracy

            print("整体测试集上的Loss：{}".format(total_test_loss))
            print("整体测试集上的正确率：{}".format(self.accuracy))
            log_text.writelines("整体测试集上的Loss：{}".format(total_test_loss))
            log_text.write('\r\n')
            log_text.writelines("整体测试集上的正确率：{}".format(self.accuracy))
            log_text.write('\r\n')
            self.writer.add_scalar("test_loss",total_test_loss,total_test_step)
            self.writer.add_scalar("test_accuracy",total_accuracy / len(test_loader.dataset),total_test_step)
            total_test_step = total_test_step + 1

            # 保存该轮模型
            torch.save(self.network, "Python/model_{}.pth".format(i))
            # torch.save(self.optimizer.state_dict(), "optimizer_{}.pth".format(i))
            print("第{}轮模型保存成功".format(i+1))
            log_text.writelines("第{}轮模型保存成功".format(i)) 
        print("----------模型训练结束-----------")     
        log_text.writelines("----------模型训练结束-----------")
        log_text.write('\r\n')
        log_text.close()
```

### 4.5 模型训练（main_MNIST.py）

前面我们已经定义好了训练所使用的相关模块，下面我们我们调用相应的依赖项以及文件进行模型的训练。

首先导入相关的模块：

```py
import torch
import matplotlib.pyplot as plt
```

接下来引入我们前面构建的类：

```python
from dataloader import Dataloader
from net import Net
from trainer import Trainer
```

然后定义模型训练的相关设置参数:

```python
'''---------设置相关训练参数---------'''
n_epochs = 1             # 训练轮次
batch_size_train = 8   # 训练集batchSize
batch_size_test = 8  # 测试集batchSize
learning_rate = 0.01    # 学习率
momentum = 0.5          # 动量、冲量
log_interval = 100       # 日志打印间隔次数
random_seed = 1         # 随机种子数
```

读取本地训练集以及测试集：

```python
'''---------初始化训练集和测试集---------'''
# CPU和GPU设置随机种子
torch.manual_seed(random_seed)
# 初始化数据加载
dataloader = Dataloader(batch_size_train = batch_size_train, batch_size_test = batch_size_test)
# 定义训练集加载器
train_loader = dataloader.train_loader()
# 定义测试集加载器
test_loader = dataloader.test_loader()
```

我们可以读取数据集中的几张图片，来查看我们所要使用的数据集：

```python
'''---------查看数据集---------'''
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
print(example_data[0][0].size())
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
```

最后我们定义网络，并将其网络训练器中进行训练。

```python
# 定义网络
network = Net()
# 设置工作模式
if torch.cuda.is_available():
    network = network.cuda()
# 加载模型训练器
trainer = Trainer(network, learning_rate, log_interval)
# 开始模型训练
trainer.tarin(epoch = n_epochs, train_loader = train_loader, test_loader = test_loader)
```

模型训练完后，模型文件保存到``	Python``文件夹下，日志文件保存到``	Python/logs_train``文件夹下。
