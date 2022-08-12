import torchvision
from torch.utils.data import DataLoader
# 数据集加载类
'''
    实现数据集本地读取，并将数据集按照指定要求进行预处理；
    后续模型训练直接调用DataLoader逐步进行训练
'''
class Dataset():
    def __init__(self,batch_size_train,batch_size_test):
        # 初始化训练集bath size
        self.batch_size_train = batch_size_train
        # 初始化测试集bath size
        self.batch_size_test = batch_size_test

    # 加载训练集
    def train_loader(self):
        # 调用pytorch自带的DataLoader方法加载MNIST训练集
        train_load = DataLoader(
            torchvision.datasets.MNIST('./Datasets/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
            batch_size = self.batch_size_train, shuffle=True)
        return train_load
    # 加载测试集
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