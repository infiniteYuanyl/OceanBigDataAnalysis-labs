# coding=utf-8
import argparse
from mlp import MLP
from cnn import CNN
import numpy as np

#python train.py --model mlp --dataset minst --epochs 30 --hid_c 512 256 128 64 64 32 32 16 16 --save mlp.npz --init normal
def evaluate(mlp):
    pred_results = np.zeros([mlp.test_data.shape[0]])
    for idx in range(mlp.test_data.shape[0]//mlp.batch_size):
        batch_images = mlp.test_data[idx*mlp.batch_size:(idx+1)*mlp.batch_size, :-1]
        prob = mlp.forward(batch_images)
        pred_labels = np.argmax(prob, axis=1)
        pred_results[idx*mlp.batch_size:(idx+1)*mlp.batch_size] = pred_labels
    if mlp.test_data.shape[0] % mlp.batch_size >0: 
        last_batch = mlp.test_data.shape[0]/mlp.batch_size*mlp.batch_size
        batch_images = mlp.test_data[-last_batch:, :-1]
        prob = mlp.forward(batch_images)
        pred_labels = np.argmax(prob, axis=1)
        pred_results[-last_batch:] = pred_labels
    accuracy = np.mean(pred_results == mlp.test_data[:,-1])
    print('Accuracy in test set: %f' % accuracy)
def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--model', help='model_type',type=str)
    parser.add_argument('--dataset', help='dataset_type',type=str)
    parser.add_argument('--epochs',default=100, help='max epochs to train',type=int)
    parser.add_argument('--hid_c',nargs='+', help='checkpoint path',type=int)
    parser.add_argument('--acfunc',help='checkpoint path',type=str)
    parser.add_argument('--save', help='save model path',type=str)
    parser.add_argument('--init', help='init_params_method',type=str)
    parser.add_argument('--ckpg', help='checkpoint path',type=str)
    
    args = parser.parse_args()
    return args
#python train.py --model cnn --dataset cifar10 --epochs 20 --save cnncifar10  --init normal --acfunc relu
if __name__ == '__main__':
    args = parse_args()

    hidden_channels = [512,256] if args.hid_c is None else args.hid_c
    num_classes = 10
    init_method='random'
    if args.model is None or args.model not in ['mlp','cnn','resnet']:
        raise TypeError('No model type has been pointed!')
    if args.dataset is None or args.dataset not in ['minst','cifar10']:
        raise TypeError('No dataset type has been pointed!')
    if args.model == 'mlp':
        mlp = MLP(input_size=784 if args.dataset=='minst' else 1024 *3 ,max_epoch=args.epochs,\
            hidden_dims = hidden_channels,num_classes=10,dataset_type=args.dataset,init_method=args.init \
            if not None else init_method)
        mlp.load_data()  
        mlp.build_model()
        mlp.init_model()
        if args.ckpg is not None:
            mlp.load_model(args.ckpg)
        mlp.train()
        if args.save is not None:
            mlp.save_model(args.save)
        mlp.evaluate()
    elif args.model == 'cnn':
        cnn = CNN(input_size=28 if args.dataset=='minst' else 32,\
            num_classes=10,max_epoch=args.epochs,dataset_type=args.dataset,init_method=args.init,ac_func=args.acfunc)
        cnn.load_data()
        cnn.build_model()
        cnn.init_model()
        cnn.train()
        cnn.evlautate()