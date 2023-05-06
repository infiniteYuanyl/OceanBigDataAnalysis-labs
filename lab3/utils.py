import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def pre_visualize(data,feature_set):
    
    
    plt.figure(figsize=(12, 9))
    for feature in feature_set:
        plt.plot(data[feature],label=str(feature))
    plt.xticks(range(0, data.shape[0], 200), data['dt_iso'].loc[::200], rotation=45)
    plt.title("Ocean Features", fontsize=18, fontweight='bold')
    plt.xlabel('Hours', fontsize=18)
    plt.ylabel('Features', fontsize=18)
    plt.grid()
    plt.legend()
    plt.show()

def post_visualize(y_train,pred_train,y_test,pred_test,\
                    time_step,unused_labels,hist_train,hist_val):
    
    ground_truth = np.concatenate([y_train,y_test])
    ground_truth_plot = np.empty_like(np.concatenate([ground_truth,\
                                                 np.zeros((time_step,1)).reshape(-1,1)]))
    ground_truth_plot[:,:]=np.nan
    ground_truth_plot[time_step:len(ground_truth)+time_step] = np.concatenate([y_train,y_test])
    ground_truth_plot[:time_step] = unused_labels
    # shift train predictions for plotting

    x_train_plot = np.empty_like(ground_truth_plot)
    x_train_plot[:, :] = np.nan
    x_train_plot[time_step:len(y_train)+time_step, :] = pred_train

    # shift test predictions for plotting
    x_test_plot = np.empty_like(ground_truth_plot)
    x_test_plot[:, :] = np.nan
    x_test_plot[len(y_train)+time_step -1:-1, :] = pred_test

    predict = np.append(x_train_plot,x_test_plot,axis=1)
    predict = np.append(predict,ground_truth_plot,axis=1)
    result = pd.DataFrame(predict)

    import plotly.express as px
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                    mode='lines',
                    name='Train prediction')))
    fig.add_trace(go.Scatter(x=result.index, y=result[1],
                    mode='lines',
                    name='Test prediction'))
    fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                    mode='lines',
                    name='Actual Value')))
    fig.update_layout(
    xaxis=dict(
        title_text='Hours/h',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Temperature/℃',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

    )



    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                xanchor='left', yanchor='bottom',
                                text='Results (LSTM)',
                                font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                                showarrow=False))
    fig.update_layout(annotations=annotations)
    fig.show()

    sns.set_style("darkgrid")
    fig = plt.figure()
    indexs = np.array(range(1,hist_train.shape[0]+1))
    ax = sns.lineplot(x=indexs,y=hist_train, label='train loss',color='royalblue',)
    ax = sns.lineplot(x=indexs,y=hist_val, label='val loss',color='tomato')
    ax.set_xlabel("Epoch", size = 14)
    ax.set_ylabel("Loss", size = 14)
    ax.set_title("Loss", size = 14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    plt.show()


def split_data(input, time_step):
    data_raw = np.array(input)
    
    data = []

    # 将data按time_step分组，data为长度为time_step的list
    labels = []
    
    #print('time_step',time_step)
    for index in range(len(data_raw) - time_step):
        data.append(data_raw[index: index + time_step])
        labels.append(data_raw[index+time_step,-1])
    #0-time_steps的数据，用于可视化
    unused_labels = np.array(data_raw[:time_step,-1]).reshape(-1,1)
	
    data = np.array(data)
    labels = np.array(labels)
    #print(type(data))  # (232, 20, 1)
    # 按照8:2进行训练集、测试集划分
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    #print('data',data.shape)
    x_train = data[:train_set_size, :, :-1]
    
    y_train = labels[:train_set_size].reshape(-1,1)
    #print('train_data',x_train.shape)
    x_test = data[train_set_size:, :,:-1]
    y_test = labels[train_set_size:].reshape(-1,1)

    return x_train, y_train, x_test, y_test,unused_labels


def load_data(file_path,data_max_len = 2100,data_start=0,feature_set=[],predict_feature=str,is_pre=False,time_steps=1000):#根据所选的特征读取data
    data = pd.read_csv(file_path)
    #给的文件就是顺序的，如果不是按照日期顺序，我们需要对他排序
    
    data = data[data_start:data_start+data_max_len]
    #print(len(data))
    data = data.sort_values('dt_iso')
    pre_feature = data[[predict_feature]]
    
    select_train_features = data[feature_set]
    
    if is_pre:
        pre_visualize(data=data,feature_set=feature_set)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    for feature in feature_set:
        select_train_features[feature] = scaler.fit_transform(select_train_features[feature].\
                                                              values.reshape(-1, 1))
    #数据和标签拼接
    result = pd.concat([select_train_features,pre_feature],axis=1)
    #print(result)
    return (split_data(result,time_steps),scaler)


