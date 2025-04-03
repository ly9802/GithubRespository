import torch
import os
import numpy as np
import pandas as pd;
from torch.utils.data import DataLoader
import torch.optim as optim
import logging
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from PIL import Image
import pandas as pd
from  torch.nn import CrossEntropyLoss,MSELoss,L1Loss
from sklearn.metrics import precision_score,accuracy_score,recall_score,roc_auc_score;
from sklearn.metrics import confusion_matrix;
from sklearn.metrics import average_precision_score,precision_recall_curve;

from dataset_new import dataset,collate_fn1, collate_fn2,Imageset
from model import resnet_model,classifier,selector,classifier1,classifier2
from Update import update,innerproduct,outputfile,regions_num_decision,probability_to_score

imsize=224;confidence_level=0.95; 
os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_resize=224;
num_randomcrop=224;
data_transforms={
    'swap': transforms.Compose([transforms.Resize((num_resize,num_resize)),transforms.CenterCrop(num_randomcrop) ]),
    'totensor': transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
cfg = {};cfg['data']="Covid19";cfg['numcls']=2;
rawdata_root="../../../dataset/Covid19Dataset/data";
train_pd=pd.read_csv("../../../dataset/Covid19Dataset/anno/train.txt",sep=",",header=None, names=['ImageName', 'label']);
test_pd=pd.read_csv("../../../dataset/Covid19Dataset/anno/test.txt",sep=",",header=None, names=['ImageName', 'label']);
covid_pd=pd.read_csv("../../../dataset/Covid19Dataset/anno/covid.txt",sep=',', header=None,names=['ImageName','label']);
normal_pd=pd.read_csv("../../../dataset/Covid19Dataset/anno/normal.txt",sep=',', header=None,names=['ImageName','label']);
noncovid_pd=pd.read_csv("../../../dataset/Covid19Dataset/anno/NonCovid19.txt",sep=',',header=None,names=['ImageName','label'])

batchsize=23;num_classes=2;
num_positive=batchsize;num_negative=batchsize;
backbone_name="resnet50"; regions_num=49;

data_set = {}
data_set['train'] = Imageset(img_dir=rawdata_root,anno_pd=train_pd,swap=data_transforms["swap"],totensor=data_transforms["totensor"],train=True);
data_set['val'] = Imageset(img_dir=rawdata_root,anno_pd=test_pd,swap=data_transforms["swap"],totensor=data_transforms["totensor"],train=False);
data_set['covid19']=Imageset(img_dir=rawdata_root,anno_pd=covid_pd,swap=data_transforms["swap"],totensor=data_transforms["totensor"],train=True);
data_set['normal']=Imageset(img_dir=rawdata_root,anno_pd=normal_pd,swap=data_transforms["swap"],totensor=data_transforms["totensor"],train=False);
data_set['nonCovid19']=Imageset(img_dir=rawdata_root,anno_pd=noncovid_pd,swap=data_transforms["swap"],totensor=data_transforms["totensor"],train=False);
dataloader={}

dataloader['train']=DataLoader(data_set['train'], batch_size=20,shuffle=True, num_workers=0, collate_fn=None);
dataloader['val']=DataLoader(data_set['val'], batch_size=20, shuffle=False, num_workers=0, collate_fn=None);

dataloader["covid19"]=DataLoader(data_set['covid19'],batch_size=batchsize,shuffle=True,num_workers=0,collate_fn=None);
dataloader["normal"]=DataLoader(data_set['normal'],batch_size=20,shuffle=False, num_workers=0,collate_fn=None);
dataloader["nonCovid19"]=DataLoader(data_set['nonCovid19'],batch_size=20,shuffle=False,num_workers=0,collate_fn=None);

net=resnet_model(backbone_name,num_classes);
last_layer=selector(net.channels,num_classes);
select=classifier1(net.channels,regions_num);
net.cuda();select.cuda();last_layer.cuda();
criteria_L1=L1Loss(reduction='mean');
criteria_MSE=MSELoss(reduction='mean');
criterion_CrossEntropy = CrossEntropyLoss()

base_lr=0.0002;

optimizer=optim.Adam(params=net.parameters(),lr=base_lr);
optimizer.add_param_group({"params":select.parameters(),"lr":base_lr*10});
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
epoch_num=10;isover=False;epoch=0;loss_epoch=10000;
while (isover==False) and (loss_epoch>0.01):
    loss_epoch=0;
    net.train(True);select.train(True);
    print("epoch no.",epoch,"-------------------- Training--------------------")
    for batch_no, data in enumerate(dataloader['covid19']):
        img_tensor, label_list=data;
        img_tensor=img_tensor.cuda();
        #labels=torch.tensor(label_list,dtype=torch.int64).cuda();
        labels=label_list.clone().long().cuda();
        img_featuremap=net(img_tensor)# shape=(batchsize, channels, w,h)
        img_featuremap2=net.featuremap; #shape=(batchsize, 256,14,14)
        batchsize, channel_num, height, width=img_featuremap.shape# resnet18:channels=512;resnet50:channels=2048
        img_featuremap_matrix=img_featuremap.contiguous().view(batchsize,channel_num,-1)# shape(batchsize, channels, w*h)
        img_featuremap_vector=img_featuremap_matrix.permute(0,2,1);# (batchsize,w*h, channels);
        regions_num=height*width;
        regions_label=[i for i in range(regions_num)];
        regions_label=torch.tensor(regions_label,dtype=torch.int64).cuda();
        loss_batch=0;
        for img_no in range(batchsize):
            img_regions_matrix=img_featuremap_vector[img_no,:,:];# img.shape(49,512)
            img_regions_score=select(img_regions_matrix);# shape(49,49)
            loss_img=criterion_CrossEntropy(img_regions_score,regions_label);# a tensor, 0 dimention, on cuda
            loss_batch=loss_batch+loss_img
        loss_epoch=loss_epoch+loss_batch.data.item();
        optimizer.zero_grad()
        loss_batch.backward();
        optimizer.step();
        print("the epoch no. is {}, the bach no. is {}, the bach loss is {}".format(epoch,batch_no,loss_batch.data.item()))    
    loss_epoch=loss_epoch/(batch_no+1);
    print("the epoch no is {}, the mean epoch loss is {} ".format(epoch, loss_epoch));
    scheduler.step();
    
    print("epoch No.",epoch,"------------------------Validation--------------------------")
    net.eval();select.eval();
    regions_label=[i for i in range(regions_num)];
    regions_label=torch.tensor(regions_label,dtype=torch.int64).cuda(); # shape (49)
    predict_epoch=None;
    for batch_no, data in enumerate(dataloader['covid19']):
        img_tensor, label_list=data;
        img_tensor=img_tensor.cuda();
        labels=label_list.clone().long().cuda();
        img_featuremap=net(img_tensor);# shape=(batchsize, channels, w,h)
        batchsize, channel_num, height, width=img_featuremap.shape
        img_featuremap_matrix=img_featuremap.contiguous().view(batchsize,channel_num,-1)
        img_featuremap_vector=img_featuremap_matrix.permute(0,2,1);
        regions_num=height*width;
        predict_batch=None;
        count=0;
        for img_no in range(batchsize):
            img_regions_matrix=img_featuremap_vector[img_no,:,:];# img.shape(49,512)
            img_regions_score=select(img_regions_matrix);
            _, pred_label_vector = torch.max(img_regions_score, dim=1);
            pred_label=pred_label_vector.contiguous().view(1,-1);
            iscorrect=torch.equal(pred_label_vector,regions_label)
            if iscorrect==True:
                count=count+1;
            else:
                pass;
            if img_no==0:
                predict_batch=pred_label;    
            else:
                predict_batch=torch.cat((predict_batch,pred_label),dim=0); 
        if batch_no==0:
            predict_epoch=predict_batch;
        else:
            predict_epoch=torch.cat((predict_epoch,predict_batch),dim=0);
        print("The epoch is {},the batch no. is {},the batch accuary is {}".format(epoch,batch_no,count/batchsize)); 
    regions_epoch=regions_label.expand_as(predict_epoch);  
    isover= torch.equal(predict_epoch,regions_epoch);
    epoch=epoch+1; # because while loop used , so i need to use epoch+  

print("-------------------- Testing--------------------");  
net.eval(); select.eval();
regions_label=[i for i in range(regions_num)];
regions_label=torch.tensor(regions_label,dtype=torch.int64).cuda(); # shape (49)
accumulation=torch.zeros_like(regions_label);
for batch_no, data in enumerate(dataloader['normal']):       
    img_tensor, label_list=data;
    img_tensor=img_tensor.cuda();

    labels=label_list.clone().long().cuda();
    img_featuremap=net(img_tensor); #shape=(batchsize, channels, w,h) 
    batchsize, channel_num, height, width=img_featuremap.shape
    img_featuremap_matrix=img_featuremap.contiguous().view(batchsize,channel_num,-1) #(batchsize, channels,widith*height)
    img_featuremap_vector=img_featuremap_matrix.permute(0,2,1);  #(batchsize,width*height, channels); 
    regions_num=height*width; 
    for img_no in range(batchsize):
        img_regions_matrix=img_featuremap_vector[img_no,:,:];# img.shape(49,512)
        img_regions_score=select(img_regions_matrix);
        _, pred_label_vector = torch.max(img_regions_score, dim=1);
        pred_label=pred_label_vector.contiguous().view(1,-1);
        indice=torch.ne(pred_label_vector,regions_label);
        onehot_vector=torch.zeros_like(regions_label)
        onehot_vector[indice]=1;
        accumulation=accumulation+onehot_vector
        a=pred_label_vector[indice];
        if a.numel()>0:
            print("the regions have:",a);
regions1=accumulation.view(1,1,height,width);
print(regions1);
num_different_points=49;
all_frequency_vector, indice=torch.topk(accumulation,k=num_different_points,dim=0,largest=True,sorted=True)

total_occurs=torch.sum(regions1);
print("the sum of all regions is:",total_occurs);
num_selected, indice2,high_frequency_vector=regions_num_decision(accumulation,confidence_level);
partial_occurs=torch.sum(high_frequency_vector);
print("The sum of high frequency erros:", partial_occurs);
print("the threshold is {}".format(torch.div(partial_occurs,total_occurs)))
print("the selected regions value:",high_frequency_vector, "\nselected regions:",indice);
print("the number of regions selected:",num_selected);
print("the regions selected:",indice2);

net.cpu();select.cpu();
torch.save(net, './net_model/selector_%f_net_%s.pt'%(confidence_level,backbone_name));
torch.save(select,'./net_model/selector_%f_fc_%s.pt'%(confidence_level,backbone_name));
indice2=indice2.cpu();
with open("./region.txt",'w') as f:
    f.write(str(num_selected)+'\n'+str(list(indice2.numpy()))+'\n')
regions_label=regions_label.cpu(); accumulation=accumulation.cpu();
high_frequency_vector=high_frequency_vector.cpu();
regions1=regions1.cpu();
for k in range(10):
    torch.cuda.empty_cache()
print("number of GPU",torch.cuda.device_count());
  
print("----------------The stage2 training---------------------------------------------");
base_lr=0.0001;momentum=0.9;threshold=0.2;
os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net2=resnet_model(backbone_name, num_classes);
indice2=indice2.to(device);
fullconnected_layer2 = classifier2(net2.channels, indice2, num_classes, 7, 7);
softmax_function =nn.Softmax(dim=1)
net2.to(device);
fullconnected_layer2.to(device);
optimizer2=optim.SGD(params=net2.parameters(),lr=base_lr,momentum=momentum);
optimizer2.add_param_group({"params":fullconnected_layer2.parameters(),"lr":base_lr*10,"momentum":momentum});
#optimizer2=optim.Adam(params=net2.parameters(),lr=base_lr);
#optimizer2.add_param_group({"params":fullconnected_layer2.parameters(),"lr":base_lr*10});

scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.1)
criterion_CrossEntropy2 = CrossEntropyLoss();

epoch=0;sensitivity=0;precision=0;recall=0; flag=True;
epoch_num=100;
for epoch in range(epoch_num):
    net2.train(True);
    fullconnected_layer2.train(True);
    print("epoch no.",epoch,"-------------------- Training--------------------");
    batch_num = len(dataloader['train'])# how many batch
    loss_epoch=0;
    for batch_no, data in enumerate(dataloader['train']):
        img_tensor, label_list=data;
        img_tensor=img_tensor.to(device);
        labels=label_list.clone().long().to(device);
        img_featuremap=net2(img_tensor)
        class_score=fullconnected_layer2(img_featuremap);# shape (20,2)
        loss_batch=criterion_CrossEntropy2(class_score,labels);
        loss_epoch = loss_epoch + loss_batch.data.item();
        optimizer2.zero_grad()
        loss_batch.backward();
        optimizer2.step();
        print("the epoch no. is {}, the bach no. is {}/{}, the bach loss is {}".format(epoch,batch_no,batch_num,loss_batch.data.item()));
    print("the epoch no. is {}, the mean loss for this epoch is {}".format(epoch, loss_epoch / (batch_no + 1)));
    scheduler.step();
    print("epoch No.",epoch,"------------------------Validation--------------------------");  
    net2.eval();fullconnected_layer2.eval();
    val_correct_num=0;predict=None;ground_true=None;
    for batch_no, data in enumerate(dataloader['val']):
        img_tensor, label_list=data;
        img_tensor=img_tensor.to(device);
        labels=label_list.clone().long().to(device);
        img_featuremap=net2(img_tensor);
        class_score=fullconnected_layer2(img_featuremap);# class_score shape(batchsize, num_classes);
        #_,predict_label=torch.max(class_score,dim=1);
        predict_probability = softmax_function(class_score).cpu().data.numpy();
        predict_label = probability_to_score(predict_probability, threshold).to(device);
        if batch_no==0:
            predict=predict_label;ground_true=labels;   
        else:
            predict=torch.cat((predict,predict_label),dim=0);
            ground_true=torch.cat((ground_true, labels),dim=0);
        batch_correct_num = torch.sum((predict_label == labels)).cpu().data.item()
        acc_batch=batch_correct_num/len(labels)
        val_correct_num += batch_correct_num;
        print("The epoch is {}, the batch no.is {}/{}, the batch accuary is {}".format(epoch,batch_no,batch_num,acc_batch)); 
    val_acc1=accuracy_score(ground_true.cpu().numpy(),predict.cpu().numpy());
    precision=precision_score(ground_true.cpu().numpy(),predict.cpu().numpy(),pos_label=0);
    recall=recall_score(ground_true.cpu().numpy(), predict.cpu().numpy(),pos_label=0);

    roc_auc=roc_auc_score(ground_true.cpu().numpy(), predict.cpu().numpy());
    true_pos, false_neg, false_pos, true_neg = confusion_matrix(ground_true.cpu().numpy(), predict.cpu().numpy()).ravel();
    if sensitivity > recall:
        with open("./net_model/performance.txt",'w') as f1:
            f1.write(str(epoch)+': '+str(sensitivity)+'\n');
    sensitivity=recall;
    specificity=recall_score(ground_true.cpu().numpy(),predict.cpu().numpy(),pos_label=1);
    average_precision=average_precision_score(ground_true.cpu().numpy(),predict.cpu().numpy());
    print("The epoch is {}, precision:{}, recall:{}, roc_auc:{}, val_acc1:{}".format(epoch, precision,recall,roc_auc,val_acc1));
    print("The epoch is {}, average precision is {}".format(epoch,average_precision));
    print("The epoch is {}, sensitivity: {}, specificity: {}\n".format(epoch,sensitivity,specificity));
    print("The Confusion Matrix as follows:\n");
    print("               predict_Covid19, predict_nonCovid\n");
    print("true_Covid19 :   {}                {}\n".format(true_pos,false_neg));
    print("true_nonCovid19: {}                {}\n".format(false_pos,true_neg));

torch.save(net2, './net_model/net_%s_epoch%d_threshold_%f.pt'%(backbone_name,epoch_num,threshold));
torch.save(fullconnected_layer2,'./net_model/fc_%s_epoch%d_threshold_%f.pt'%(backbone_name,epoch_num,threshold));

'''
print("-------------------- Testing--------------------");         
net2.eval();
fullconnected_layer2.eval();
val_correct_num=0;predict=None;ground_true=None;
length = len(dataloader['val'])
for batch_no, data in enumerate(dataloader['val']):
    img_tensor,label_list=data;
    img_tensor=img_tensor.to(device);
    labels=label_list.clone().long().to(device);
    img_fmap=net2(img_tensor);
    score_vector=fullconnected_layer2(img_fmap);
    _, pred_label = torch.max(score_vector, dim=1);
    if batch_no==0:
        predict=pred_label;ground_true=labels;   
    else:
        predict=torch.cat((predict,pred_label),dim=0);
        ground_true=torch.cat((ground_true, labels),dim=0);
    batch_correct_num = torch.sum((pred_label == labels)).data.item()
    val_correct_num += batch_correct_num;
    
    logging.info('[TEST]: Batch {:03d} / {:03d}'.format(batch_no+1, length));

val_acc1 = 1.0 * val_correct_num / len(data_set['val'])
#val_acc1=accuracy_score(ground_true.cpu().numpy(),predict.cpu().numpy());
precision=precision_score(ground_true.cpu().numpy(),predict.cpu().numpy(),pos_label=0);
recall=recall_score(ground_true.cpu().numpy(), predict.cpu().numpy(),pos_label=0);
roc_auc=roc_auc_score(ground_true.cpu().numpy(), predict.cpu().numpy());

true_pos, false_neg, false_pos, true_neg = confusion_matrix(ground_true.cpu().numpy(), predict.cpu().numpy()).ravel();

sensitivity=recall;
specificity=recall_score(ground_true.cpu().numpy(),predict.cpu().numpy(),pos_label=1);
average_precision=average_precision_score(ground_true.cpu().numpy(),predict.cpu().numpy());

if epoch==epoch_num-1:
    precision_array, recall_array,threhold_array=precision_recall_curve(ground_true.cpu().numpy(),predict.cpu().numpy(),pos_label=0)

print("The epoch is {}, precision: {}, recall: {}, roc_auc: {}, val_acc1: {}".format(epoch, \
precision,recall, roc_auc,val_acc1),flush=True);
print("The epoch is {}, average precision is {}".format(epoch,average_precision));
print("The epoch is {}, sensitivity: {}, specificity: {}\n".format(epoch,sensitivity,specificity));
print("The Confusion Matrix as follows:\n");
print("               predict_Covid19, predict_nonCovid\n");
print("true_Covid19 :   {}                {}\n".format(true_pos,false_neg));
print("true_nonCovid19: {}                {}\n".format(false_pos,true_neg));
torch.save(net2, './net_model/net_%s_epoch%d.pt'%(backbone_name,epoch_num));
torch.save(fullconnected_layer2,'./net_model/fc_%s_epoch%d.pt'%(backbone_name,epoch_num));
'''