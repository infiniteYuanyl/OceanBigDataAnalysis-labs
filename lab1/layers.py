# coding=utf-8
import numpy as np
import struct
import os
import time

class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))
    def init_param(self, std=0.01,init_method='even'):  # 参数初始化
        if init_method == 'even':
        
            print('FullyConnectedLayer initializes evenly')
            self.weight = np.random.rand(self.num_input, self.num_output)
            self.bias = np.random.rand(1, self.num_output)
        elif init_method == 'zero':
            print('FullyConnectedLayer initializes with zeros')
            self.weight = np.zeros([self.num_input, self.num_output])
            self.bias = np.zeros([1, self.num_output])
        elif init_method == 'normal':
            
            print('FullyConnectedLayer initializes normly')
            self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
            self.bias = np.zeros([1, self.num_output])
        else:
            raise NotImplementedError('The init_method: %s are not implentmented!'%(init_method))
    def forward(self, input):  # 前向传播计算
        start_time = time.time()
        self.input = input
        self.output = np.dot(self.input,self.weight) + self.bias
        
        return self.output
    def backward(self, top_diff):  # 反向传播的计算
        self.d_weight = np.dot(self.input.T,top_diff)
        self.d_bias = np.sum(top_diff,axis=0)
        bottom_diff = np.dot(top_diff,self.weight.T)
        
        return bottom_diff
    def update_param(self, lr):  # 参数更新
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr *  self.d_bias
        
    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self):  # 参数保存
        return self.weight, self.bias

class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input
        output = self.input * (self.input > 0)
        return output
    def backward(self, top_diff):  # 反向传播的计算
        #同理，把那些input > 0的位置全部 * 1（relu的求导），原来被抑制为0的位置不变，不传播损失。
        
        bottom_diff = top_diff * ( self.input >0 )
        
        return bottom_diff

class SigmoidLayer(object):
    def __init__(self):
        print('\tSigmoid layer.')
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input
        
        output = 1 / (1 + np.exp(-self.input ))
        self.output = output
        return output
    def backward(self, top_diff):  # 反向传播的计算
        
        # sigmoid 导数为sig(x) * (1 - sig(x))
        bottom_diff = top_diff * self.output * (1-self.output)
        
        return bottom_diff

class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')
    def forward(self, input):  # 前向传播的计算
        
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp,axis=1,keepdims=True).reshape(-1,1)
        # print('softmax predict',self.prob)
        return self.prob
    def get_loss(self, label):   # 计算损失
        label = label.astype(np.int32)
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    def backward(self):  # 反向传播的计算
        # TODO：softmax 损失层的反向传播，计算本层损失
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

class DropoutLayer(object):
    def __init__(self,p=0.3):  
        self.p =p     
        print('\tDropout layer.')
    def forward(self, input):  # 前向传播的计算
        self.input = input
        np.random.seed(1234)
        self.mask = (np.random.rand(self.input.shape[0],self.input.shape[1]) >=self.p / (1 - self.p))    
        
        self.input = self.input * self.mask
        output = self.input
        return output
    def backward(self, top_diff):  # 反向传播的计算
        
        bottom_diff = top_diff * self.mask 
        return bottom_diff


    
class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01,init_method='even'):
        if init_method == 'even':
            # print('randn ok')
            print('ConvolutionalLayer initializes evenly')
            self.weight = np.random.rand(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)
            self.bias = np.random.rand(self.channel_out)
        elif init_method == 'zero':
            print('ConvolutionalLayer initializes with zeros')
            self.weight = np.zeros([self.channel_in, self.kernel_size, self.kernel_size, self.channel_out])
            self.bias = np.zeros([self.channel_out])
        elif init_method == 'normal':
            # print('normal ok')
            print('ConvolutionalLayer initializes normly')
            self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
            self.bias = np.zeros([self.channel_out])
        else:
            raise NotImplementedError('The init_method: %s are not implentmented!'%(init_method))
        
    def forward(self, input):
        # 把卷积核转换为行向量，对应局部数据hs：hs+k，ws：ws+k转化成列向量
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        # cin,k,k,cout -> cin*k*k,cout
        self.weight_reshape = np.reshape(self.weight,(-1,self.channel_out))
        # img2col:N * H * W,cin*k*k
        
        self.img2col = np.zeros( [self.input.shape[0]*height_out*width_out ,self.channel_in*self.kernel_size*self.kernel_size ])
        
        # 将img转换为列向量
        for idxn in range(self.input.shape[0]):
            for idxh in range(height_out):
                for idxw in range(width_out):
                    #计算出转换后的行索引
                    row_i = idxn*height_out*width_out+idxh*width_out+idxw                
                    # img2:cin*k*k,input_pad:cin,k,k,reshape:cin,k,k->cin*k*k
                    self.img2col[row_i,:]= self.input_pad[idxn,:,idxh*self.stride:idxh*\
                            self.stride+self.kernel_size,idxw*self.stride:idxw*self.stride+self.kernel_size].reshape([-1])
        #output = img2col x weight :N * H * W ,cout
        output = np.dot(self.img2col ,self.weight_reshape) + self.bias
        self.output = output.reshape([self.input.shape[0],height_out,width_out,-1]).transpose([0,3,1,2])
        
        self.forward_time = time.time() - start_time
        return self.output
    def backward(self, top_diff):
        
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        self.d_bias = np.sum(top_diff,axis=(2,3))
        bottom_diff = np.zeros(self.input_pad.shape)
        self.col2img = np.zeros([self.input_pad.shape[0]*self.input_pad.shape[2]*self.input_pad.shape[3]\
             ,self.channel_in*self.kernel_size*self.kernel_size ])
        # print(self.col2img.shape)
        # padding
        self.d_bias = np.sum(top_diff, axis=(0, 2, 3))
        input_pad = np.zeros([self.input_pad.shape[0], self.input_pad.shape[1], \
            self.input_pad.shape[2]+2*self.kernel_size-2, self.input_pad.shape[3]+2*self.kernel_size-2])
        input_pad[:, :, self.kernel_size-1:1-self.kernel_size, self.kernel_size-1:1-self.kernel_size] = self.input_pad
        # top_diff->top_diff_pad_reshape:N ,cout,H , W => N * H * W ,cout
        top_diff_pad = np.zeros([top_diff.shape[0], top_diff.shape[1], top_diff.shape[2]+2*self.kernel_size-2, top_diff.shape[3]+2*self.kernel_size-2])
        top_diff_pad[:, :, self.kernel_size-1:1-self.kernel_size, self.kernel_size-1:1-self.kernel_size] = top_diff
        top_diff_pad_reshape = np.zeros([self.input.shape[0]*self.input_pad.shape[2]*self.input_pad.shape[3], top_diff.shape[1]*(self.kernel_size**2)])
        # print(top_diff_pad_reshape.shape)
        # print(top_diff.shape)
        # print(self.input_pad.shape)
        for idxn in range(self.input_pad.shape[0]):       
            for idxh in range(self.input_pad.shape[2]):
                for idxw in range(self.input_pad.shape[3]):
                    row_i = idxn * self.input_pad.shape[2] * self.input_pad.shape[3] + idxh * self.input_pad.shape[3] + idxw
                    top_diff_pad_reshape[row_i,:] = top_diff_pad[idxn, :,\
                         idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size].reshape([-1]) 
                         
                    self.col2img[row_i,:] = input_pad[idxn, :,\
                         idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size].reshape([-1])
        
        
        
        self.d_weight = np.dot(self.col2img.T,np.sum(top_diff_pad_reshape.reshape([top_diff_pad_reshape.shape[0],\
            self.channel_out,self.kernel_size,self.kernel_size]),axis=(2,3))).reshape(self.channel_in,self.kernel_size,self.kernel_size,-1)
        print('dweight',self.d_weight)
        bottom_diff = np.matmul(top_diff_pad_reshape , np.rot90(self.weight, k=2, axes=(1,2)).transpose(3,1,2,0).reshape(-1, self.channel_in))  
        bottom_diff = np.reshape(bottom_diff,[self.input_pad.shape[0],self.input_pad.shape[2],self.input_pad.shape[3],self.channel_in]).transpose(0,3,1,2)
        bottom_diff = bottom_diff[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]]
        
        
        
        return bottom_diff
    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def get_forward_time(self):
        return self.forward_time
    def get_backward_time(self):
        return self.backward_time

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    
    def forward(self, input):
        start_time = time.time()
        
        self.input = input
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        self.input_col = np.zeros([self.input.shape[0], self.input.shape[1], height_out*width_out, self.kernel_size**2])
        for i in range(height_out):
            for j in range(width_out):
                self.input_col[:,:,i*width_out+j,:] = self.input[:, :, i*self.stride:i*self.stride+self.kernel_size, j*\
                    self.stride:j*self.stride+self.kernel_size].reshape(self.input.shape[0], self.input.shape[1], -1)
        self.output_col = np.max(self.input_col, axis=3)
        max_index_col = np.zeros([self.input.shape[0]*self.input.shape[1]*height_out*width_out, self.kernel_size**2])
        max_index_col[np.arange(self.input.shape[0]*self.input.shape[1]*height_out*width_out), np.argmax(self.input_col, axis=3).reshape(-1)] = 1
        max_index_col = max_index_col.reshape([self.input.shape[0], self.input.shape[1], height_out*width_out, self.kernel_size**2])
        self.output = self.output_col.reshape([self.input.shape[0], self.input.shape[1], height_out, width_out])
        self.max_index = np.zeros(self.input.shape)
        for i in range(height_out):
            for j in range(width_out):
                self.max_index[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]\
                     = max_index_col[:,:,i*width_out+j,:].reshape([self.input.shape[0], self.input.shape[1], self.kernel_size, self.kernel_size])

        return self.output
    def backward(self, top_diff):

        bottom_diff = np.multiply(self.max_index, top_diff.repeat(self.kernel_size,2).repeat(self.kernel_size,3))
        return bottom_diff
class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):
        # print(input.shape)
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output
    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        # b,c,h,w = self.input.shape
        # top_diff = top_diff.reshape(b,c,h,w)
        
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff
