# Modify from offical 
# Authod: IconBall
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import torchvision
from torchvision import datasets, transforms
import time
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)
from PIL import Image
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset(dir,fold,dataset=None,list_file='./list.csv'):
    data=pd.read_csv(list_file)
    data['code']=data.label.astype('category').cat.codes
    if (fold is not None)&(dataset!='test'):
        data=data[data.fold!=fold].reset_index(drop=True)
        sss=StratifiedShuffleSplit(n_splits=1,test_size=0.125,random_state=1024)
        train,val=list(sss.split(data,data.code))[0]
        if dataset=='train':
            data=data[data.index.isin(train)]
        elif dataset=='val':
            data=data[data.index.isin(val)]
    elif (fold is not None)&(dataset=='test'):
        data=data[data.fold==fold]
        
    images=[]
    files=list(os.walk(dir))[0][2]
    files.sort()
    for sample in files:
        #extention=sample.split('.')[1]
        file = int(sample.split('_')[0])
        if file in data.file_name.tolist():
            label = data[data.file_name==file].code.item()
            path=os.path.join(dir,sample)
            item=(path,label)
            images.append(item)
            
    return images


class DatasetFolder(object):
    def __init__(self, root,list_file,loader=pil_loader,transform=None,fold=None,dataset='train'):
        samples = make_dataset(root,list_file=list_file,fold=fold,dataset=dataset)
        self.root = root
        self.loader = loader
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path,idx= self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample,torch.tensor(np.int32(idx),dtype=torch.int64)

    def __len__(self): 
        return len(self.samples)
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True
            
def getModel(model_name,num_feats,pretrained=True):
     
    if model_name=='resnet50':
        model=torchvision.models.resnet50(pretrained=pretrained)
        model.fc=nn.Linear(model.fc.in_features,num_feats)
        setattr(model,'input_size',(3,224,224))     
        
    if model_name=='InceptionV4':
        model=inv4.InceptionV4(num_classes=num_feats)
        setattr(model,'input_size',(3,224,224))
        
    if model_name=='VGG16':
        model=torchvision.models.vgg16_bn(pretrained=pretrained)
        seq=list(model.classifier.children())[:-1]+[torch.nn.Linear(4096,num_feats)]
        model.classifier=torch.nn.Sequential(*seq)
        setattr(model,'input_size',(3,224,224))
        
    if model_name=='xception':
        model=xc.Xception(num_classes=num_feats)
        setattr(model,'input_size',(3,224,224))
        
    if model_name=='inceptionresnetv2':
        model=InceptionResNetV2(num_classes=num_feats)
        setattr(model,'input_size',(3,224,224))
    return model

def train_model(model, dataloaders, criterion, params_to_update,num_epochs=25, is_inception=False,best_acc=0,early_stop_round=5):
    lr=0.045
    optimizer=optim.SGD(params_to_update,lr=lr,momentum=0.9)   
    since = time.time()
    epoch_acc=0
    val_loss_history = [0]
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter=0
    last_epoch_acc=0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
        #for phase in [ 'val','train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            dataphase=tqdm(dataloaders[phase])
            for inputs, labels in dataphase:
                if phase=='val':
                    dataphase.set_description(f"[{MODEL_NAME}][{epoch}] evaluating")
                else:
                    dataphase.set_description(f"[{MODEL_NAME}][{epoch}][{best_acc:.4f}][{last_epoch_acc:.4f}]")
                inputs = inputs.to(device)
                labels = labels.to(device)
                #labels = torch.LongTensor(labels[:,0]).view(-1).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception becaudataloadersse in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)    #torchvision.transforms.RandomCrop(224),(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_acc))
            #if epoch_acc>=0.99:
               #return model
            if phase == 'val':
                print(f"[{MODEL_NAME}][{epoch}][{best_acc:.4f}][{epoch_acc:.4f}]")
                val_loss_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    early_stop_counter=0
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(),f'{MODEL_NAME}_fold_{NUM_FOLD}_best.pth')
                else:
                    early_stop_counter+=1
                last_epoch_acc=epoch_acc
        if early_stop_counter>=early_stop_round:
            break
        if (epoch>0)&(epoch%2==0):
            lr=lr*0.94  
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    return model



if __name__=='__main__':
    
    NUM_FEATS=11
    NUM_EPOCHS=50
    NUM_FOLD=1
    MODEL_NAME='resnet50'
    PRETRAINED=True
    
    
    is_inception=True if MODEL_NAME=='InceptionV3' else False
    model = getModel(MODEL_NAME,NUM_FEATS,PRETRAINED)
    train_transform=torchvision.transforms.Compose([
        #torchvision.transforms.Resize((224,224)),
	#torchvision.transforms.RandomResizedCrop((224,224),(0.8,1.2),(1,1)),
        #torchvision.transforms.Resize((input_size,input_size)),
        torchvision.transforms.RandomOrder([
            torchvision.transforms.RandomRotation((-30,30)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomGrayscale(p=1),
        ]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x:x*2-1),
    ])

    val_transform=torchvision.transforms.Compose([
        #torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x:x*2-1),
    ])

    train=DatasetFolder('./data/',transform=train_transform,fold=NUM_FOLD,dataset='train',list_file='list.csv')
    val=DatasetFolder('./data/',transform=val_transform,fold=NUM_FOLD,dataset='val',list_file='list.csv')    
    trainloader = torch.utils.data.DataLoader(train,batch_size=64, shuffle=True)
    evalloader = torch.utils.data.DataLoader(val,batch_size=64, shuffle=False)
    dataloaders_dict={'train':trainloader,'val':evalloader}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Send the model to GPU
    model = model.to(device)
    model.train()
    params_to_update = model.parameters()
    print("Params to learn:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
    criterion=nn.CrossEntropyLoss(reduction='mean')
    # Train and evaluate
    model = train_model(model, dataloaders_dict, criterion, params_to_update,num_epochs=NUM_EPOCHS, is_inception=is_inception,best_acc=0,early_stop_round=20)
    torch.save(model.state_dict(),f'{MODEL_NAME}_fold_{NUM_FOLD}_final.pth')
