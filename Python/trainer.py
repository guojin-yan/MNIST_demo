# 使用tensorboard页面，方便观测训练过程中的Loss
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import time


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

