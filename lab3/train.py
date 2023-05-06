# coding=utf-8
import os
import sys

import time
import pandas as pd
import argparse
from utils import load_data,post_visualize,pre_visualize
from models import LSTM
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--epochs', help='epochs',type=int,default=150)
    parser.add_argument('--in_dims', help='input_dims',type=int,default=6)
    parser.add_argument('--hid_dims', help='hid_dims',type=int,default=64)
    parser.add_argument('--recurrent_layers', help='layer',type=int,default=4)
    parser.add_argument('--out_dims', help='out_dims',type=int,default=1)
    parser.add_argument('--time_steps', help='time_steps',type=int,default=100)
    parser.add_argument('--features_select', help='features',nargs='+')
    parser.add_argument('--feature_pre', help='pre_feature',type=str,default='temp')
    parser.add_argument('--data_path', help='data_path',type=str,default='data/weather.csv')
    parser.add_argument('--data_start', help='data_start_idx',type=int,default=0)
    parser.add_argument('--data_len', help='data_max_items',type=int,default=2600)
    parser.add_argument('--save', help='the dir to save models')


    args = parser.parse_args()
    return args

#python train.py --features_select temp_min temp_max pressure humidity wind_speed wind_deg --in_dims 6
if __name__ == '__main__':
    args = parse_args()
    
    input_dim = args.in_dims
    # 隐藏层特征的维度
    hidden_dim = args.hid_dims
    # 循环的layers
    num_layers = args.recurrent_layers
    # 预测后一段时刻到参数
    output_dim = args.out_dims
    num_epochs = args.epochs
    feature_set = args.features_select
    feature_pre = args.feature_pre
    data_path = args.data_path
    data_start = args.data_start
    data_max_len = args.data_len
    time_steps = args.time_steps
    print(data_path)
    device = torch.device('mps:0')
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,betas=(0.9, 0.99))
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    
    data_all ,scaler= load_data(file_path=data_path,data_max_len=data_max_len,data_start=data_start,\
                                feature_set=feature_set,is_pre=False,predict_feature=feature_pre,time_steps=time_steps)
    x_train,y_train,x_test,y_test,unused_labels = data_all
    # print(y_train.shape)
    
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    # print(x_train.device)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train_labels= torch.from_numpy(y_train).type(torch.Tensor)
    y_test_labels= torch.from_numpy(y_test).type(torch.Tensor)
    #print(x_train.shape)

    # dataset = Data.TensorDataset(x_train, y_train_labels)
    # dataloader = Data.DataLoader(
    # # 从数据库中每次抽出batch size个样本
    #                     dataset=dataset,
    #                     batch_size=100,
    #                     shuffle=False,
    #                     num_workers=4,
    #                     drop_last=True)

    hist_train = np.zeros(num_epochs)
    hist_val = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []
    # print('y_train',y_train)
    # print('y_train_shape',y_train.shape)
    
    #model.train()
    for t in range(num_epochs):

        y_preds = model(x_train.to(device))
        #y_preds = model(x_train)
        loss = criterion(y_preds, y_train_labels.to(device))
        #loss = criterion(y_preds.reshape(-1,1), y_train_labels.reshape(-1,1))
        print("Epoch ", t, "MSE: ", loss.item())
        hist_train[t] =loss.item()
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_preds_val = model(x_test.to(device))
            loss_val = criterion(y_preds_val,y_test_labels.to(device))
            hist_val[t] = loss_val.item()
        
        
    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))
    y_test_preds = model(x_test.to(device)).cpu().detach().numpy()
    y_preds = y_preds.cpu().detach().numpy()


    post_visualize(y_train,y_preds,y_test,y_test_preds,time_step=time_steps,\
                    unused_labels=unused_labels,hist_train=hist_train,hist_val=hist_val)



