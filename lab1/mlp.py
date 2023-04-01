# coding=utf-8
import numpy as np
import struct
import os
import time
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE_DIR)

from layers import FullyConnectedLayer, ReLULayer, SigmoidLayer,SoftmaxLossLayer,DropoutLayer
from data_load import load_minst,load_cifar10
MNIST_DIR = "../data/mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


class BaseLayer(object):
    def __init__(self, in_channel, out_channel, drop_out=False,use_ac_func=True,use_relu=True,\
        init_method='random'):
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.fc = FullyConnectedLayer(in_channel,out_channel)
        self.ac_func = None if not use_ac_func else (ReLULayer() if use_relu else SigmoidLayer())
        self.use_ac = use_ac_func
        self.init_method = init_method
        
    def load_param(self, weight, bias):
        self.fc.load_param(weight, bias)
    def init_param(self):
        print(self.init_method)
        self.fc.init_param(init_method=self.init_method)
    def save_param(self):  # 参数保存
        return self.fc.save_param()
    def update_param(self, lr):  # 参数更新
        self.fc.update_param(lr)
    def forward(self,input):
        x = self.fc.forward(input)
        if self.use_ac:      
            x = self.ac_func.forward(x)
        return x
    def backward(self,bottom_diff):
        dloss = bottom_diff
        if self.use_ac:   
            dloss = self.ac_func.backward(dloss)
        dloss = self.fc.backward(dloss)   
        
        return dloss


class MLP(object):
    def __init__(self, batch_size=100, input_size=784,hidden_dims=[],ac_func='relu',init_method='random',\
     num_classes=10, lr=0.005, max_epoch=1, print_iter=100,dataset_type='cifar10'):
        self.batch_size = batch_size
        self.input_size = input_size
        self.in_channels = hidden_dims
        self.num_classes = num_classes
        self.lr = lr
        self.ac_func = ac_func
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.init_method = init_method
        self.dataset_type = dataset_type

    def load_data(self):
        
        if self.dataset_type == 'cifar10':
            print('Loading CIFAR10 data from files...')
            train_images ,train_labels ,test_images,test_labels = load_cifar10()
        elif self.dataset_type == 'minst':
            print('Loading MNIST data from files...')
            train_images ,train_labels ,test_images,test_labels = load_minst()
        
        if self.dataset_type=='cifar10':
            train_images = train_images.reshape(train_images.shape[0],-1)
            test_images = test_images.reshape(test_images.shape[0],-1)
            train_labels = train_labels.reshape(train_images.shape[0],1)
            test_labels = test_labels.reshape(test_images.shape[0],1)
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)
        

    
        
    def shuffle_data(self):
        print('Randomly shuffle data...')
        np.random.shuffle(self.train_data)
    
    def build_model(self):  # 建立网络结构
        
        print('Building multi-layer perception model...')
        self.layers = []
        self.in_channels = [self.input_size] + self.in_channels + [self.num_classes]
        use_ac_func = True
        for i in range(len(self.in_channels) - 1):
            if i == len(self.in_channels) - 2:
                use_ac_func = False
            self.layers.append(BaseLayer(self.in_channels[i],self.in_channels[i+1],use_ac_func=use_ac_func,\
                use_relu = True if self.ac_func == 'relu' else False, init_method=self.init_method))
            
        self.softmax = SoftmaxLossLayer()
        

    def init_model(self):
        print('Initializing %s parameters of each layer in MLP...'%(self.init_method))
        for layer in self.layers:
            layer.init_param()

    def load_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        params = np.load(param_dir).item()
        for i,layer in enumerate(self.layers):
            layer.load_param(params['w'+str(i+1)], params['b'+str(i+1)])

    def save_model(self, param_dir):
        print('Saving parameters to file ' + param_dir)
        params = {}
        for i,layer in enumerate(self.layers):
            params['w'+str(i+1)], params['b'+str(i+1)] = layer.save_param()
        np.save(param_dir, params)

    def forward(self, input):  # 神经网络的前向传播
        x = input
        for i,layer in enumerate(self.layers):
            x = layer.forward(x)
        prob = self.softmax.forward(x) 
        return prob
   

    def backward(self):  # 神经网络的反向传播

        dloss = self.softmax.backward()
        for layer in reversed(self.layers):
            dloss = layer.backward(dloss)
        return dloss
            

    def update(self, lr):
        for layer in self.layers:
            layer.update_param(lr)

    def train(self):
        max_batch = self.train_data.shape[0] // self.batch_size
        
        print('Start training...')
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)             
                self.backward()       
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(self.test_data.shape[0]/self.batch_size):
            batch_images = self.test_data[idx*self.batch_size:(idx+1)*self.batch_size, :-1]
            start = time.time()
            prob = self.forward(batch_images)
            end = time.time()
            print("inferencing time: %f"%(end-start))
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:,-1])
        print('Accuracy in test set: %f' % accuracy)

    
