
# coding=utf-8
import numpy as np
import struct
import os
import time
import pickle

def normalized(data,std=0.5):
    min_vals = np.min(data, axis=1)
    max_vals = np.max(data, axis=0)
    
    # 归一化到 [0, 1] 区间
    data_norm = (data - min_vals) / (max_vals - min_vals)
    
    # 计算每一列的标准差
    std_vals = np.std(data_norm, axis=0)
    
    # 标准化为标准差为 0.5
    data_norm = std * (data_norm / std_vals)
    return data_norm
def load_minst(with_hw=False):
    MNIST_DIR = "../data/mnist_data"
    TRAIN_DATA = "train-images-idx3-ubyte"
    TRAIN_LABEL = "train-labels-idx1-ubyte"
    TEST_DATA = "t10k-images-idx3-ubyte"
    TEST_LABEL = "t10k-labels-idx1-ubyte"
    train_images = load_mnist_data(os.path.join(MNIST_DIR, TRAIN_DATA), True)
    train_labels = load_mnist_data(os.path.join(MNIST_DIR,TRAIN_LABEL),False)
    test_images = load_mnist_data(os.path.join(MNIST_DIR,TEST_DATA),True)
    test_labels = load_mnist_data(os.path.join(MNIST_DIR,TEST_LABEL),False)
    # train_images = train_images.astype(np.float32) / 255.0
    # test_images = test_images.astype(np.float32) / 255.0
    # print('train: ',train_images[2,:])
    if with_hw:
        train_images = train_images.reshape(-1,28,28)
        test_images = test_images.reshape(-1,28,28)
    return train_images,train_labels,test_images,test_labels
    
def load_cifar10():
    data_dir = '../data/cifar10'
    train_images = np.zeros((50000, 32, 32, 3), dtype=np.uint8)
    train_labels = np.zeros((50000,), dtype=np.uint8)
    test_images = np.zeros((10000, 32, 32, 3), dtype=np.uint8)
    test_labels = np.zeros((10000,), dtype=np.uint8)

    for i in range(1, 6):
        data_dict = unpickle(data_dir + '/data_batch_{}'.format(i))
        train_images[(i-1)*10000:i*10000, :, :, :] = np.transpose(np.reshape(data_dict[b'data'], (10000, 3, 32, 32)), (0, 2, 3, 1))
        train_labels[(i-1)*10000:i*10000] = data_dict[b'labels']

    data_dict = unpickle(data_dir + '/test_batch')
    test_images[:, :, :, :] = np.transpose(np.reshape(data_dict[b'data'], (10000, 3, 32, 32)), (0, 2, 3, 1))
    test_labels[:] = data_dict[b'labels']
    # train_images = train_images.astype(np.float32) / 255.0
    # test_images = test_images.astype(np.float32) / 255.0

    # for i in range(3):
    #     train_images[:, :, i] = (train_images[:, :, i] - 0.5) / 0.5
    #     test_images[:, :, i] = (test_images[:, :, i] - 0.5) / 0.5

    # Split the dataset into training and testing sets
    num_train = int(train_images.shape[0] * 0.8)
    train_images, val_images = train_images[:num_train], train_images[num_train:]
    train_labels, val_labels = train_labels[:num_train], train_labels[num_train:]
    
    return train_images, train_labels, test_images, test_labels

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def load_mnist_data(file_dir, is_images = 'True'):
    
    # Read binary data
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    # Analysis file header
    if is_images:
        # Read images
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    else:
        # Read labels
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1
    data_size = num_images * num_rows * num_cols
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
    mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
    print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
    return mat_data