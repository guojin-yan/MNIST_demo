import torchvision
from torch.utils.data import DataLoader
 
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

    # 加载训练集
    def train_loader(self):
        # 调用pytorch自带的DataLoader方法加载MNIST训练集；
        # 直接使用pytorch的MNIST数据集API接口加载数据，download可以设置是否下载数据集，
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