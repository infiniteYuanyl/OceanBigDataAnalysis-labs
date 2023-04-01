
# coding:utf-8
import numpy as np
import struct
import os

import time

from mlp import MLP
from layers import FullyConnectedLayer, ReLULayer, SigmoidLayer,SoftmaxLossLayer ,ConvolutionalLayer, MaxPoolingLayer, FlattenLayer
from data_load import load_cifar10,load_minst
class CNN(object):
    def __init__(self,batch_size=2000,lr = 0.01,max_epoch=10,input_size=28,\
        num_classes=10,conv_layers=[],pool_layers=[],fc_dims=[],print_iter=1,dataset_type='minst',init_method='random',ac_func='relu'):
        self.input_size = input_size
        self.max_epoch = max_epoch
        self.lr = lr
        self.conv_layers = conv_layers
        self.pool_layers = pool_layers
        self.init_method = init_method
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.ac_func = ac_func
        self.dataset_type = dataset_type
        self.print_iter = print_iter
        self.param_layer_name = []

    def build_model(self):
        print('Building CNN...')

        self.channel_in = 1 if self.dataset_type == 'minst' else 3
        self.channel_out = 16
        self.conv_layers = [[3,self.channel_in,6,1,1],[3,6,self.channel_out,1,1]]
        self.pool_layers = [[2,2],[2,2]]

        self.layers = {}
        layer_depth = 0
        for i,(conv_layer,pool_layer) in enumerate(zip(self.conv_layers,self.pool_layers)):
            self.layers['conv_'+str(i+1)] = ConvolutionalLayer(kernel_size=conv_layer[0], channel_in=conv_layer[1], \
                channel_out=conv_layer[2], stride=conv_layer[3], padding=conv_layer[4])
            self.param_layer_name.append('conv_'+str(i+1))
            if self.ac_func == 'relu':
                self.layers['relu_'+str(i+1)] = ReLULayer()
                self.param_layer_name.append('relu_'+str(i+1))
            elif self.ac_func == 'sigmoid':
                self.layers['sigmoid_'+str(i+1)] = SigmoidLayer()
                self.param_layer_name.append('sigmoid_'+str(i+1))
            else:
                raise NotImplementedError('activate function %s has not been implmented!'%(self.ac_func))
            self.layers['pool_'+str(i+1)] = MaxPoolingLayer(kernel_size=pool_layer[0],stride=pool_layer[1])
            self.param_layer_name.append('pool_'+str(i+1))
            layer_depth = layer_depth + 1
        
        
        output_size = self.input_size//(2**layer_depth)
        self.conv_pool_output_shape = [self.channel_out,output_size,output_size]
        self.layers['flatten'] = FlattenLayer(self.conv_pool_output_shape,[self.channel_out*output_size*output_size])
        self.param_layer_name.append('flatten')
        fc_dims = [120,84]
        self.layers['mlp'] = MLP(input_size = self.channel_out*output_size*output_size,\
            hidden_dims=fc_dims,init_method=self.init_method,ac_func=self.ac_func,\
            num_classes= self.num_classes)
        self.layers['mlp'].build_model()
        self.layers['mlp'].init_model()
        self.param_layer_name.append('mlp')           
        

        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name:
                self.update_layer_list.append(layer_name)

    def init_model(self):
        print('Initializing parameters of each layer...')
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param(init_method = self.init_method)

    def load_image(self, image_dir, image_height, image_width):
        print('Loading and preprocessing image from ' + image_dir)
        self.input_image = scipy.misc.imread(image_dir)
        image_shape = self.input_image.shape
        self.input_image = scipy.misc.imresize(self.input_image,[image_height, image_width,3])
        self.input_image = np.array(self.input_image).astype(np.float32)
        self.input_image -= self.image_mean
        self.input_image = np.reshape(self.input_image, [1]+list(self.input_image.shape))
        # input dim [N, channel, height, width]
        self.input_image = np.transpose(self.input_image, [0, 3, 1, 2])
        return self.input_image, image_shape

    def update(self, lr):
        for idx in range(len(self.param_layer_name)):
            if 'conv' in self.param_layer_name[idx] :
                self.layers[self.param_layer_name[idx]].update_param(lr)
            elif self.param_layer_name[idx] == 'mlp':
                self.layers[self.param_layer_name[idx]].update(lr)

    def forward(self, input_image):
        start_time = time.time()
        current = input_image
        for idx in range(len(self.param_layer_name)):
            # print(self.param_layer_name[idx])
            current = self.layers[self.param_layer_name[idx]].forward(current)
            
        #print('Forward time: %f' % (time.time()-start_time))
        return current

    def backward(self):
        
        layer_idx = list.index(self.param_layer_name,'mlp')
        for idx in range(layer_idx, -1, -1):
            if self.param_layer_name[idx] =='mlp':
                dloss = self.layers[self.param_layer_name[idx]].backward()
            else:  
                dloss = self.layers[self.param_layer_name[idx]].backward(dloss)
        return dloss
    
    def train(self):
        max_batch = self.train_data.shape[0] // self.batch_size
            
        print('Start training...')
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            train_data = self.data_transform(self.train_data[:,:-1],(28,28,1) if self.dataset_type=='minst' else (32,32,3))
            labels = self.train_data[:,-1]
            for idx_batch in range(max_batch):
                batch_images = train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, :]
                batch_labels = labels[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size]
                prob = self.forward(batch_images)
                loss = self.layers['mlp'].softmax.get_loss(batch_labels)             
                self.backward()       
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))
    def evaluate(self):
        test_data = self.data_transform(self.test_data[:,:-1])
        labels = self.test_data[:,-1]
        pred_results = np.zeros([test_data.shape[0]])
        for idx in range(test_data.shape[0]/self.batch_size):
            batch_images = test_data[idx*self.batch_size:(idx+1)*self.batch_size,:]
            start = time.time()
            prob = self.forward(batch_images)
            end = time.time()
            print("inferencing time: %f"%(end-start))
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == labels)
        print('Accuracy in test set: %f' % accuracy)
    def data_transform(self,data,shape=(28,28,1)):
        h,w,cin  = shape
        return data.reshape(data.shape[0],h,w,cin).transpose([0,3,1,2])

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


        


