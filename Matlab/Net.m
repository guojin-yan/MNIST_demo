function lgraph = Net()
lgraph = layerGraph();
tempLayers = [
%     imageInputLayer([28 28 1],"Name","imageinput")
%     convolution2dLayer([1 10],5,"Name","conv_1","Padding","same")
%     batchNormalizationLayer
%     maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same")
%     reluLayer("Name","relu_1")
%     convolution2dLayer([10 20],5,"Name","conv_2","Padding","same")
%     batchNormalizationLayer
%     dropoutLayer(0.5,"Name","dropout")
%     
%     maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same")
%     reluLayer("Name","relu_2")
%     fullyConnectedLayer(320,"Name","fc_1")
%     batchNormalizationLayer
%     fullyConnectedLayer(50,"Name","fc_2")
%     reluLayer("Name","relu_3")
%     dropoutLayer(0.5,"Name","dropout_2")
%     fullyConnectedLayer(10,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")


%     imageInputLayer([28 28 1])
%     
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer

    imageInputLayer([28 28 1])
    
    convolution2dLayer(5,10,"Name","conv_1","Padding","same")
    batchNormalizationLayer
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same")
    reluLayer("Name","relu_1")
    convolution2dLayer(5,20,"Name","conv_2","Padding","same")
    batchNormalizationLayer
    dropoutLayer(0.5,"Name","dropout_1")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same")
    reluLayer("Name","relu_2")

    fullyConnectedLayer(50,"Name","fc_1")
    reluLayer("Name","relu_3")
    dropoutLayer(0.5,"Name","dropout_2")
    fullyConnectedLayer(10,"Name","fc_2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")
    ];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

end