import torch
import matplotlib.pyplot as plt
from dataloader import Dataloader
from net import Net
from trainer import Trainer

'''---------设置相关训练参数---------'''
n_epochs = 1             # 训练轮次
batch_size_train = 8   # 训练集batchSize
batch_size_test = 8  # 测试集batchSize
learning_rate = 0.01    # 学习率
momentum = 0.5          # 动量、冲量
log_interval = 100       # 日志打印间隔次数
random_seed = 1         # 随机种子数

'''---------初始化训练集和测试集---------'''
# CPU和GPU设置随机种子
torch.manual_seed(random_seed)
# 初始化数据加载
dataloader = Dataloader(batch_size_train = batch_size_train, batch_size_test = batch_size_test)
# 定义训练集加载器
train_loader = dataloader.train_loader()
# 定义测试集加载器
test_loader = dataloader.test_loader()

'''---------查看数据集---------'''

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)

# fig = plt.figure()
# print(example_data[0][0].size())
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


# 定义网络
network = Net()
# 设置工作模式
if torch.cuda.is_available():
    network = network.cuda()
# 加载模型训练器
trainer = Trainer(network, learning_rate, log_interval)
# 开始模型训练
trainer.tarin(epoch = n_epochs, train_loader = train_loader, test_loader = test_loader)

