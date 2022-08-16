clear all;

%% 获取数据集
[train_X, train_Y, test_X, test_Y] = Dataloader();


%% 定义网络

layers = Net();

%% 设置网络训练配置
max_epochs = 3; % 训练轮次
min_batchSize = 8; % 分块尺寸
options = trainingOptions('adam', ...   % 求解器设定为ADAM，亚当（Adam）是神经网络算法的优化求解器
    'ExecutionEnvironment','gpu', ...   % 选择运行设备，gpu or cpu
    'MaxEpochs',max_epochs, ...         % 最大训练周期数
    'GradientThreshold',1, ...          % 梯度临界点,防止梯度爆炸
    'InitialLearnRate',0.01, ...       % 初始学习率
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize',min_batchSize, ...  % 分块尺寸
    'LearnRateDropPeriod',10, ...      % 学习率下降开始次数
    'LearnRateDropFactor',0.2, ...      % 200轮之后乘以0.2开始降低学习率
    'Verbose',0, ...
    'Plots','training-progress' ...     % 绘制训练过程
    );


%% 网络训练
%doTraining = false;
doTraining = true;
if doTraining
    net = trainNetwork(train_X,train_Y,layers,options);
else
    load('MNIST_clas.mat','net');
end

%% 模型保存
save('MNIST_clas.mat','net')
