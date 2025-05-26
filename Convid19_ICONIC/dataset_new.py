import os
import torch
from torch.utils import data
from PIL import Image
import pandas as pd;
class Imageset(data.Dataset):
    def __init__(self,img_dir=None,anno_pd=None,swap=None,totensor=None,train=False):
        self.paths=anno_pd['ImageName'].tolist();
        self.root_path=img_dir;
        self.labels=anno_pd['label'].tolist();
        self.swap=swap;
        self.totensor=totensor;
        self.train=train;
    def pil_loader(self,img_path):
        #str img_path image path
        return Image.open(img_path).convert('RGB');
    def __len__(self):
        return len(self.paths)
    def __getitem__(self,item):
        img_path=os.path.join(self.root_path,self.paths[item]);
        img=self.pil_loader(img_path);
        img_swap=self.swap(img);
        img_tensor=self.totensor(img_swap);
        label = self.labels[item];
        #label_tensor=torch.tensor([int(label)],dtype=torch.int64);
        return img_tensor,label
class dataset(data.Dataset):
    def __init__(self, cfg, imgroot,anno_pd, stage=0,num_positive=0, num_negative=0, swap=None,totensor=None, train=False):
        super(dataset,self).__init__();
        self.root_path=imgroot;
        self.paths=anno_pd['ImageName'].tolist();
        self.labels=anno_pd['label'].tolist();
        self.swap=swap;
        self.totensor= totensor;
        self.anno_pd=anno_pd;
        self.cfg=cfg;
        self.num_positive=num_positive;
        self.num_negative=num_negative;
        self.train=train;
    def __len__(self):
        return len(self.paths);
    def __getitem__(self, item):
        img_path=os.path.join(self.root_path,self.paths[item]);
        img=self.pil_loader(img_path);
        img_swap=self.swap(img);
        img_swap=self.totensor(img_swap);
        label = self.labels[item];
        if self.train:
            positive_images=self.fetch_positive(1,label,self.paths[item])
            negative_images=self.fetch_negative(1,label,self.paths[item])
            return img_swap, positive_images,negative_images, label
            # negative_images : list, len(list)=batchsize
            #tuple is the value returned
        return img_swap, label
    def fetch_positive(self, num, label, path):

        other_img_info = self.anno_pd[(self.anno_pd.label == label) & (self.anno_pd.ImageName != path)]
        other_img_info = other_img_info.sample(min(num, len(other_img_info))).to_dict('records')
        other_img_path = [os.path.join(self.root_path, e['ImageName']) for e in other_img_info]
        other_img = [self.pil_loader(img) for img in other_img_path]
        other_img_swap=[self.swap(img) for img in other_img]
        other_img_swap = [self.totensor(img) for img in other_img_swap]
        return other_img_swap
    def fetch_negative(self, num, label, path):
        negative_img_info=self.anno_pd[(self.anno_pd.label!=label)&(self.anno_pd.ImageName!=path)]
        negative_img_info=negative_img_info.sample(min(num, len(negative_img_info))).to_dict(orient='records')
        negative_img_path=[os.path.join(self.root_path, e['ImageName']) for e in negative_img_info]
        negative_img=[self.pil_loader(img) for img in negative_img_path]
        negative_img_swap=[self.swap(img) for img in negative_img]
        negative_img_swap=[self.totensor(img) for img in negative_img_swap]
        return negative_img_swap
    def pil_loader(self,img_path):
        #str img_path image path
        return Image.open(img_path).convert('RGB');
def collate_fn1(batch):
    '''
    is used for trainingdataset
    '''
    imgs = []
    postive_imgs = []
    negative_imgs=[]
    labels = []
    for sample in batch:
        # tuple(tensor, list[tensor], int) sample 
        # tensor sample[0]; list[tensor] sample[1]; list[tensor] sample[2];int sample[3]=labels
        imgs.append(sample[0])# the first is an image, it is a tensor
        postive_imgs.extend(sample[1])# the second is a list
        negative_imgs.extend(sample[2])# the third is a list, so "extend" function is used
        labels.append(sample[3])
    #print(torch.stack(negative_imgs,0).shape);
    return torch.stack(imgs, 0), torch.stack(postive_imgs, 0), torch.stack(negative_imgs,0), labels

def collate_fn2(batch):
    '''
    it is used for testing dataset
    '''
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
    return torch.stack(imgs, 0), label
