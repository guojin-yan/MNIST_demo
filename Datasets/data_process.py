import numpy as np
import struct
 
from PIL import Image
import os

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
# 读取过程与读取图片一致
label_data_buffer = open(label_file, 'rb').read()
magic, numLabels = struct.unpack_from('>II', label_data_buffer, 0)
label_datas = struct.unpack_from('>' + label_data_size, label_data_buffer, struct.calcsize('>II'))
label_datas = np.array(label_datas).astype(np.int64)


'''如果想将图片文件保存到本地，可以运行以下代码'''

# train_path = os.path.join(home_path, 'mnist_train_image')   # 转换后的训练集所在路径
	
# # 创建 0~9 十个文件夹，存放转换后的图片
# for i in range(10): 
#     file_name = train_path + os.sep + str(i)
#     if not os.path.exists(file_name):
#         os.mkdir(file_name)
 
# for ii in range(numLabels):
#     imges = Image.fromarray(image_datas[ii, 0, 0:28, 0:28])
#     label = label_datas[ii]
#     file_name = train_path + os.sep + str(label) + os.sep + str(ii) + '.png'
#     imges.save(file_name)
