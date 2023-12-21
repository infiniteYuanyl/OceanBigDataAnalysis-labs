import numpy as np
import datetime
import matplotlib.pyplot as plt
setting = 'informer_OCEAN_ftMS_sl64_ll32_pl16_dm256_nh8_el2_dl1_df1024_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'
# exp = Exp_Informer(args)
# exp.predict(setting, True)

def draw_loss_graph(train_loss,val_loss,test_loss):
    plt.figure()
    plt.plot(train_loss,label='Train Loss')
    plt.plot(val_loss,label='Validation Loss')
    plt.plot(test_loss,label='Test Loss')
    plt.xlabel('Train Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
preds = np.load('result/pred.npy')
trues = np.load('result/true.npy')
 
print(trues.shape)
print(preds.shape)
start_date = datetime.datetime(1950, 5, 14, 20)
end_date = datetime.datetime(1950, 6, 15, 19)
hours = int((end_date - start_date).total_seconds() / 3600) + 1
date_times = [start_date + datetime.timedelta(hours=h) for h in range(hours)]
plt.figure()
plt.plot(date_times,trues[:,0,-1].reshape(-1),label='GroundTruth')
plt.plot(date_times,preds[:,0,-1].reshape(-1),label='Prediction')
plt.legend()
plt.show()
data = np.load('result/loss0.npz')
train_loss = data['train']
val_loss = data['val']

test_loss = data['test']
draw_loss_graph(train_loss,val_loss,test_loss)


