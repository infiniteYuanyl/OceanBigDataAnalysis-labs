# -*- coding: utf-8 -*-
import netCDF4 as nc
import os
from netCDF4 import Dataset
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
def np2rgbimg(path,data,labels,resize_shape):
    b,h,w = data.shape
    data_path = path + '/train'
    label_path = path + '/masks'
    rgbdata = np.repeat(data,3,axis=0).reshape(b,3,h,w).transpose(0,2,3,1)
    rgbmasks = np.repeat(labels,3,axis=0).reshape(b,3,h,w).transpose(0,2,3,1)
    
    for i in range(b):
        filename = 'zos_cglo_'+ str(i) + '.png'
        image_array = Image.fromarray(rgbdata[i,:,:])
        img= T.Resize(resize_shape)(image_array)
        img.save(data_path + filename)
        
        labels_array = Image.fromarray(rgbmasks[i,:,:])
        label= T.Resize(resize_shape)(labels_array)
        label.save(label_path + filename)
    print("imgs and labels has been saved in {%s}.",path)


if __name__ == '__main__':
    nc_obj=Dataset('data/ssha.nc')

    #查看nc文件有些啥东东
    print(nc_obj)
    print('---------------------------------------')

    #查看nc文件中的变量
    print(nc_obj.variables.keys())
    for i in nc_obj.variables.keys():
        print(i)
    # print('---------------------------------------')
    # print(nc_obj.variables['time'])
    # print('---------------------------------------')
    # print(nc_obj.variables['latitude'])
    # print('---------------------------------------')
    # print(nc_obj.variables['longitude'])
    print('---------------------------------------')
    print(nc_obj.variables['zos_cglo'])
    b,h,w = nc_obj.variables['zos_cglo'][:,:].shape
    data = np.zeros((b,h,w))
    labels = np.zeros((b,h,w))
    raw_data = nc_obj.variables['zos_cglo'][:,:]
    data = ((raw_data - np.mean(raw_data)) / np.std(raw_data) + 1)/2
    data = data * 255
    labels[np.argwhere(raw_data - np.mean(raw_data) > 0.5)] = 1
    labels[np.argwhere(raw_data - np.mean(raw_data) < -0.5)] = 2

    print('data',data[:2,:,:].shape)
    print('labels:',labels[:2,:,:])
    base_path = 'data/'
    np2rgbimg(base_path,data,labels,(512,512))


    
