import numpy as np
import torch
from torch import nn
from  torchvision import models, transforms, datasets
import torch.nn.functional as F
class resnet_model(nn.Module):
    def __init__(self,modelname,num_classes):
        super(resnet_model,self).__init__();
        self.modelname=modelname;
        if modelname=="resnet50":
           model=models.resnet50(pretrained=True);
           self.channels=2048;
        elif modelname=="resnet18":
           model=models.resnet18(pretrained=True);
           self.channels=512;
        else:
            raise NotImplementedError("No such stage")
        self.net=nn.Sequential(*list(model.children())[:-2]);
        self.net2=nn.Sequential(*list(model.children())[:-3]);
        self.avgpool=model.avgpool;
        self.fc=nn.Linear(self.channels, num_classes);
        self.featuremap=None;
    def forward(self, x):
        x2=self.net(x);
        self.featuremap=self.net2(x);
        return x2;
class classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(classifier,self).__init__();
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1));
        self.fc=nn.Linear(in_features=in_features,out_features=num_classes,bias=False);
    def forward(self,x):
        #x1=x[:,:,1:4,1:3];# x1 shape=(batchsize,2048,3,2);
        #x2=x[:,:,1:4,4:6];# x2 shape=(batchsize,2048,3,2);
        #x3=torch.cat((x1,x2),dim=-1);# x3 shape=(batchsize,2048,3,4)
        x4=self.avgpool(x);
        batchsize=x4.size(0);
        x5=x4.contiguous().view(batchsize,-1);
        out=self.fc(x5)
        #x_length = torch.norm(x5, p=2, dim=-1,keepdim=True);# (batchsize,1)
        #weight=self.fc.weight.data
        #weight_norm=torch.norm(weight,p=2,dim=1,keepdim=True).transpose(1,0);#(num_classes,1);
        #matrix=torch.mm(x_length,weight_norm)# matrix:(batchsize,num_classes);
        #result=torch.div(out,matrix,rounding_mode='true');
        #result = torch.div(out,matrix);
        return out
class classifier1(nn.Module):
    def __init__(self,in_features,num_regions):
        super(classifier1,self).__init__();
        self.fc=nn.Linear(in_features=in_features,out_features=num_regions,bias=False);

    def forward(self,x):
        return self.fc(x);
class classifier2(nn.Module):
    def __init__(self,in_features,indice,num_classes,height, width):
        # int in_features
        # tensor indice
        # int num_classes;
        super(classifier2,self).__init__();
        self.indice=indice;
        self.num_classes=num_classes;
        self.in_features=in_features;
        self.fc=nn.Linear(in_features=in_features,out_features=num_classes,bias=False);
        sample = torch.zeros((width, height)).contiguous().view(-1);
        sample[self.indice] = torch.tensor([1], dtype=torch.float32);
        self.temp_one=sample;
    def forward(self,x):
        # x should be a feature map which has batchsize, channels, w, h
        batchsize, channels, h, w=x.shape;
        x1=x.contiguous().view(batchsize,channels, -1);
        x1=x1.permute(0,2,1);# tensor , shape should be (batchsize, h*w, channels)
        x2=torch.index_select(input=x1,dim=1, index=self.indice); #shape(batchszie, topk, channels)
        x3=torch.mean(x2,dim=1,keepdim=False);# x3 shape (batchsize, channels); channels=self.in_features
        x4=self.fc(x3);
        return x4;

class selector(nn.Module):
    def __init__(self, num,num_classes):
        super(selector,self).__init__();
        self.select=nn.Linear(in_features=num, out_features=num_classes,bias=False);
    def forward(self,x):
        return self.select(x);