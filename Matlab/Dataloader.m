%% 数据加载器方法
% 返回
function [train_X, train_Y,test_X, test_Y] = Dataloader()
fid = fopen('dataset\t10k-labels-idx1-ubyte', 'rb');
% 读取测试集标签
test_labels = fread(fid, inf, 'uint8', 'l');
test_labels = test_labels(9:end);
fclose(fid);
test_Y = categorical(test_labels);
% 读取训练集标签
fid = fopen('dataset\train-labels-idx1-ubyte', 'rb');
train_labels = fread(fid, inf, 'uint8', 'l');
train_labels = train_labels(9:end);
fclose(fid);
train_Y = categorical(train_labels);

% 读取训练集图像数据
fid = fopen('dataset\train-images-idx3-ubyte', 'rb');
train_images_data = fread(fid, inf, 'uint8', 'l');
train_images_data = train_images_data(17:end);
train_images_data = mapminmax(train_images_data',0,1)';
fclose(fid);
train_images = reshape(train_images_data,28,28,60000);
train_images = permute(train_images,[2 1 3]);
train_X(:,:,1,:) = train_images;

% 读取测试集图像数据
fid = fopen('dataset\t10k-images-idx3-ubyte', 'rb');
test_images_data = fread(fid, inf, 'uint8', 'l');
test_images_data = test_images_data(17:end);
fclose(fid);
test_images = reshape(test_images_data,28,28,10000);
test_images = permute(test_images,[2 1 3]);
test_X(:,:,1,:) = test_images;
end