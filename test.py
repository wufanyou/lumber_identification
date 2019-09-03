import torch
import argparse
import torch.nn as nn
import numpy as np
import copy
import torchvision
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import time
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
from utils import *

def evalModel(model,evalloader,prob=False):
    with torch.set_grad_enabled(False):
        result=[]
        for x,_ in tqdm(evalloader):
            x=x.to(DEVICE)
            out=model(x)
            out=out.argmax(1) if prob is False else out
            result.append(out.cpu().detach().numpy())
    result=np.concatenate(result)
    return result

def TTA_transformer(HFP=1,VFP=1,GP=1):
    val_transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=HFP),
        torchvision.transforms.RandomVerticalFlip(p=VFP),
        torchvision.transforms.RandomGrayscale(p=GP),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x:x*2-1),
        ])
    return val_transform

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_NAME',type=str, default='resnet50')
    parser.add_argument('--NUM_FOLD',type=int, default=0)
    parser.add_argument('--BATCH_SIZE',type=int,default=32)
    parser.add_argument('--TTA', action='store_true')
    parser.add_argument('--LUMBER', action='store_true')
    parser.add_argument('--CPU', action='store_true')
    
    args = parser.parse_args()
    
    NUM_FEATS = 11
    FOLDS = [args.NUM_FOLD] if args.NUM_FOLD in range(5) else range(5)
    MODEL_NAME = args.MODEL_NAME
    IS_TTA = args.TTA
    BATCH_SIZE = args.BATCH_SIZE
    IS_LUMBER = args.LUMBER
    DEVICE = 'cpu' if args.CPU else 'cuda:0'
    ALL_WEIGHT= {'resnet50':0.94792,'densenet121':0.93519,'mobilenet_v2':0.92895}
    
    if not IS_TTA:
        val_transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Lambda(lambda x:x*2-1),
                        ])
        if MODEL_NAME in ['resnet50','densenet121','mobilenet_v2']:
            model = getModel(MODEL_NAME,NUM_FEATS,False).to(DEVICE).eval()
            arr = []
            cm = np.zeros([11,11])
            for NUM_FOLD in FOLDS:
                model.load_state_dict(torch.load(f'{MODEL_NAME}_fold_{NUM_FOLD}_best.pth'))
                val = DatasetFolder('./data/',transform=val_transform,fold=NUM_FOLD,dataset='val',list_file='list.csv')
                evalloader = torch.utils.data.DataLoader(val,batch_size=BATCH_SIZE, shuffle=False)
                result = evalModel(model,evalloader)
                result = pd.DataFrame({'Predicted':result,'True':[ x[1] for x in val.samples],'file':[x[2] for x in val.samples]})
                
                if IS_LUMBER:
                    counter=0
                    for _,x in result.groupby('file'):
                        x=x.reset_index(drop=True)
                        counter+=int(x.Predicted.mode()[0]==x['True'][0])
                    arr.append(counter/len(result.file.unique()))
                    
                else:
                    arr.append((result['Predicted']==result['True']).mean())
                cm += confusion_matrix(result['True'],result['Predicted'])
            arr = np.array(arr)    
            print(f'mean {arr.mean():.5f} std:{arr.std():.5f}')
            cm = np.round(cm/cm.sum(0),3).T
            print(cm)
            print(cm.sum(1))
           
            
        elif MODEL_NAME in ['all','ALL']:
            arr = []
            for NUM_FOLD in FOLDS:
                result=0
                val = DatasetFolder('./data/',transform=val_transform,fold=NUM_FOLD,dataset='test',list_file='list.csv')
                evalloader = torch.utils.data.DataLoader(val,batch_size=BATCH_SIZE, shuffle=False)
                for MODEL_NAME in ['resnet50','densenet121','mobilenet_v2']:
                    model = getModel(MODEL_NAME,NUM_FEATS,False).to(DEVICE).eval()
                    model.load_state_dict(torch.load(f'{MODEL_NAME}_fold_{NUM_FOLD}_best.pth'))
                    result += evalModel(model,evalloader,True)*ALL_WEIGHT[MODEL_NAME]
                result=result.argmax(1)
                result = pd.DataFrame({'Predicted':result,'True':[ x[1] for x in val.samples],'file':[x[2] for x in val.samples]})
                    
                if IS_LUMBER:
                    counter=0
                    for _,x in result.groupby('file'):
                        x=x.reset_index(drop=True)
                        counter+=int(x.Predicted.mode()[0]==x['True'][0])
                    arr.append(counter/len(result.file.unique()))
                    
                else:
                    arr.append((result['Predicted']==result['True']).mean())
                    
            arr = np.array(arr)
            print(f'mean {arr.mean():.5f} std:{arr.std():.5f}')
    else:
        arr = []
        model = getModel(MODEL_NAME,NUM_FEATS,False).to(DEVICE).eval()
        for NUM_FOLD in FOLDS:
            result=0
            val=DatasetFolder('./data/',fold=NUM_FOLD,dataset='test',list_file='list.csv')
            for T in range(8):
                model.load_state_dict(torch.load(f'{MODEL_NAME}_fold_{NUM_FOLD}_best.pth'))
                val.transform=TTA_transformer(T%2,T//2%2,T//2//2%2)
                evalloader = torch.utils.data.DataLoader(val,batch_size=BATCH_SIZE, shuffle=False)
                result += evalModel(model,evalloader,True)*ALL_WEIGHT[MODEL_NAME]
            result=result.argmax(1)
            result = pd.DataFrame({'Predicted':result,'True':[ x[1]for x in val.samples]})    
            arr.append((result['Predicted']==result['True']).mean())
        arr = np.array(arr)
        print(f'mean {arr.mean():.5f} std:{arr.std():.5f}')
    