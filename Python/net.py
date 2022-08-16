import torch.nn as nn
import torch.nn.functional as F
 
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
    def forward(self, x):
        # 一次卷积运算
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # 二次卷积运算
        x = self.conv2(x)
        x = self.conv2_drop(x, 2)
        x = F.max_pool2d(x)
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