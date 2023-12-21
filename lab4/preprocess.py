# -*- coding: utf-8 -*-
import netCDF4 as nc

import matplotlib.pyplot as plt
import os
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import datetime
from vmdpy import VMD
if __name__ == '__main__':
    nc_obj=Dataset('data/sea_temperature_water_velocity.nc')

    # #查看nc文件中的变量
    #print(nc_obj.variables.keys())
    print(nc_obj['time'])
    print('---------------')
    #print(nc_obj['depth'])
    
    raw_data = nc_obj['thetao_cglo'][:,:]
    data = np.mean(raw_data,axis=1).reshape(-1,1)
    norm_data = ((data - np.mean(data)) / np.std(data) + 1) / 2

    # 生成日期时间序列
    N = norm_data.shape[0]
    start_date = datetime.datetime(1950, 1, 1, 0)
    hours = np.arange(N).reshape(-1, 1)
    date_times = [start_date + datetime.timedelta(hours=int(h)) for h in hours]

    # 分解模态数量
    num_modes = 7
    # # 创建DataFrame对象并合并日期时间列
    modes,_,_ = VMD(norm_data.squeeze(), alpha=2000,tau=0,tol=1e-6,K=num_modes,DC=0,init=1)
    

    # # 创建DataFrame对象并合并日期时间列和VMD模态
    df = pd.DataFrame(date_times, columns=['date'])
    for i, mode in enumerate(modes):
        df[f'mode {i+1}'] = mode
    df['ori'] = data
    df['temp'] = data


    # 写入CSV文件
    df.to_csv('informer/data/temp0.csv', index=False)
    